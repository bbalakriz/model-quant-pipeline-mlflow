# quantization_pipeline.py
from kfp import compiler, dsl
from kfp.dsl import InputPath, OutputPath
from kfp.kubernetes import (
    add_toleration,
    CreatePVC,
    DeletePVC,
    mount_pvc,
    use_secret_as_env,
)

MOUNT_POINT = "/models"
BASE_MODEL_PATH = MOUNT_POINT + "/base-model"
OPTIMIZED_MODEL_PATH = MOUNT_POINT + "/optimized-model"


@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        'huggingface-hub',
        #'boto3',
    ]
)
def download_model(
    model_id: str,
    output_path: str,
):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=model_id, local_dir=output_path)
    print("Model downloaded successfully from HF.")


@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        'llmcompressor==0.6.0',
        'transformers==4.52.2',
        'accelerate',
        'vllm'
    ]
)
def quantize_model(
    model_path: str,
    output_path: str,
    quantization_type: str,
):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # 1) Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2) Data calibration
    from datasets import load_dataset

    # This is hardcoded but it could be parametrized in the pipeline
    NUM_CALIBRATION_SAMPLES = 256  # 1024
    DATASET_ID = "neuralmagic/LLM_compression_calibration"
    DATASET_SPLIT = "train"

    # Load dataset.
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    ds = ds.shuffle(seed=42).select(range(NUM_CALIBRATION_SAMPLES))

    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        return {"text": example["text"]}
    ds = ds.map(preprocess)

    # Tokenize the data
    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            truncation=False,
            add_special_tokens=True,
        )
    ds = ds.map(tokenize, remove_columns=ds.column_names)

    # 3) Quantize model
    from llmcompressor.modifiers.quantization import GPTQModifier
    from llmcompressor.transformers import oneshot
    from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # This is hardcoded but it could be parametrized in the pipeline
    DAMPENING_FRAC = 0.1  # 0.01
    OBSERVER = "mse"  # minmax
    GROUP_SIZE = 128  # 64
    # Configure the quantization algorithm to run.
    ignore=["lm_head"]
    mappings=[
        [["re:.*q_proj", "re:.*k_proj", "re:.*v_proj"], "re:.*input_layernorm"],
        [["re:.*gate_proj", "re:.*up_proj"], "re:.*post_attention_layernorm"],
        [["re:.*down_proj"], "re:.*up_proj"]
    ]
    if quantization_type == "int8":
        recipe = [
            SmoothQuantModifier(smoothing_strength=0.7, ignore=ignore, mappings=mappings),
            GPTQModifier(
                targets=["Linear"],
                ignore=ignore,
                scheme="W8A8",
                dampening_frac=DAMPENING_FRAC,
                observer=OBSERVER,
            )
        ]
    elif quantization_type == "int4":
        recipe = [
            GPTQModifier(
                targets=["Linear"],
                ignore=ignore,
                scheme="w4a16",
                dampening_frac=DAMPENING_FRAC,
                observer=OBSERVER,
                group_size=GROUP_SIZE
            )
        ]
    elif quantization_type == "fp8":
        # Configuring simple PTQ quantization for fp8; 
        recipe = [
            QuantizationModifier(
                targets="Linear",
                scheme="FP8_DYNAMIC",
                ignore=ignore
            )
        ]
    else:
        raise ValueError(f"Quantization type {quantization_type} not supported")

    # pass the right set of params for oneshot
    # simple PTQ (fp8 dynamic) does not require calibration data for weight quantization
    def init_oneshot():
        if quantization_type in ["int8", "int4"]:
            return oneshot(
                    model=model,
                    dataset=ds,
                    recipe=recipe,
                    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
                    max_seq_length=8196,
                )
        elif quantization_type == "fp8":
            return oneshot(
                    model=model,
                    recipe=recipe,
                    max_seq_length=8196,
                )
        else:
            raise ValueError(f"Unsupported quantization_type: {quantization_type}")

    init_oneshot()
    # Save to disk compressed.
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)


@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        'boto3',
    ]
)
def upload_model(
    model_path: str,
    s3_path: str,
):
    import os
    from boto3 import client

    print('Starting results upload.')
    s3_endpoint_url = os.environ["s3_host"]
    s3_access_key = os.environ["s3_access_key"]
    s3_secret_key = os.environ["s3_secret_access_key"]
    s3_bucket_name = os.environ["s3_bucket"]

    print(f'Uploading predictions to bucket {s3_bucket_name} '
          f'to S3 storage at {s3_endpoint_url}')

    s3_client = client(
        's3', endpoint_url=s3_endpoint_url, aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key, verify=False
    )

    # Walk through the local folder and upload files
    for root, dirs, files in os.walk(model_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            s3_file_path = os.path.join(s3_path, local_file_path[len(model_path)+1:])
            s3_client.upload_file(local_file_path, s3_bucket_name, s3_file_path)
            print(f'Uploaded {local_file_path}')

    print('Finished uploading results.')


@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        'lm_eval==v0.4.3',
        'vllm',
    ]
)
def evaluate_model(
    model_path: str,
    eval_tasks: str,
    results_json: OutputPath()
):
    """ Command to execute:
    lm_eval --model vllm \
      --model_args pretrained=$MODEL_PATH,add_bos_token=true \
      --trust_remote_code \
      --tasks eval_tasks \
      --num_fewshot 5 \
      --limit 250 \
      --batch_size 'auto'
      --output_path "results_json"
    """
    import subprocess
    import os

    model_args = "pretrained=" + model_path  + ",add_bos_token=true"

    # Execute the huggingface_hub-cli command
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    result = subprocess.run(["lm_eval",
                             "--model", "vllm",
                             "--model_args", model_args,
                             "--trust_remote_code",
                             "--tasks", eval_tasks,
                             "--num_fewshot", "5",
                             "--limit", "250",
                             "--batch_size", "auto",
                             "--output_path", results_json],
                            capture_output=True, text=True, env=env)
    # Check for errors or output
    if result.returncode == 0:
        print("Model evaluated successfully:")
        print(result.stdout)
    else:
        print("Error evaluating the model:")
        print(result.stderr)


@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        'mlflow',
        'pandas'
    ]
)
def log_to_mlflow(
    results_json: InputPath(),
    mlflow_tracking_uri: str,
    model_id: str,
    quantization_type: str
):
    import json
    import pandas as pd
    import mlflow
    import glob, os
    import re
    from datetime import datetime
    import glob

    matches = glob.glob(os.path.join(results_json, "**", "*.json"), recursive=True)
    if not matches:
        raise RuntimeError("No JSON found")

    results_json_file = matches[0]
    print(f"[DEBUG] Using JSON file for MLflow logging: {results_json_file}")


    def generate_run_name(model_id: str, quant_type: str) -> str:
        safe_model_id = re.sub(r'\W', '_', str(model_id))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{safe_model_id}_{quant_type}_{timestamp}"

    experiment_name = generate_run_name(model_id, quantization_type)

    with open(results_json_file) as f:
        data = json.load(f)

    # Flatten numeric metrics dynamically
    flat = []
    for task, metrics in data.get("results", {}).items():
        for key, value in metrics.items():
            if key == "alias":
                continue
            try:
                numeric = float(value)
            except (ValueError, TypeError):
                continue
            metric, variant = key.split(",", 1)
            flat.append({
                "task": task,
                "metric": metric,
                "variant": variant,
                "value": numeric
            })

    df = pd.DataFrame(flat)

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_artifact(results_json, artifact_path="lm_eval_json")

        for _, row in df.iterrows():
            name = f"{row['task']}_{row['metric']}_{row['variant']}"
            mlflow.log_metric(name, row["value"])

        # save and log a CSV summary
        tmp_csv = "/tmp/lm_eval_metrics.csv"
        df.to_csv(tmp_csv, index=False)
        mlflow.log_artifact(tmp_csv, artifact_path="lm_eval_metadata")


# ---------------------------
# New DSPy-based evaluation component
# ---------------------------
@dsl.component(
    base_image='registry.access.redhat.com/ubi9/python-312',
    packages_to_install=[
        # core libs: transformers & llmcompressor and evaluation libs
        'llmcompressor==0.6.0',
        'transformers==4.52.2',
        'datasets',
        'sentence-transformers',
        'dspy',
        'mlflow',
        'torch'
    ]
)
def dspy_evaluate_and_log(
    base_model_path: str,
    quantized_model_path: str,
    mlflow_tracking_uri: str,
    dspy_dataset_name: str = "xsum",
    dspy_dataset_split: str = "validation[:512]",
    dspy_max_samples: int = 512,
    mlflow_experiment_name: str = "quantization_dspy_eval",
):
    """
    Loads:
      - base model (HF checkpoint) from base_model_path
      - quantized model from LLM Compressor compressed tensors (quantized_model_path)
    Runs evaluation on a summarization dataset (default XSum) and logs:
      - embedding similarity (base_output vs reference)
      - embedding similarity (quant_output vs reference)
      - embedding similarity (quant_output vs base_output)
    All metrics are logged to MLflow and a sample CSV of comparisons is saved to artifacts.
    """

    import os
    import json
    import mlflow
    import csv
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer, util

    # Load quantized model via llmcompressor loader
    # Note: llmcompressor may expose different helper functions depending on version.
    # Try known import paths; if yours differs, please adapt.
    quant_model = None
    quant_tokenizer = None
    try:
        # Preferred loader if provided by llmcompressor 0.6.0
        from llmcompressor.loaders import load_compressed_model
        quant_model, quant_tokenizer = load_compressed_model(quantized_model_path, device_map="auto")
    except Exception:
        try:
            # Alternative import path (older/newer versions)
            from llmcompressor import load_compressed_model
            quant_model, quant_tokenizer = load_compressed_model(quantized_model_path, device_map="auto")
        except Exception as e:
            raise RuntimeError(f"Could not import llmcompressor loader to open compressed model: {e}")

    # Load base model (HF checkpoint)
    # If base_model_path is an HF ID this will work; if it's a local snapshot, it also works.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    base_model.eval()

    # Bring quant model to device if loader returned CPU model
    try:
        quant_model.to(device)
        quant_model.eval()
    except Exception:
        # some loader wrappers already put model on device or return proxy; ignore
        pass

    # Prepare dataset
    ds = load_dataset(dspy_dataset_name, split=dspy_dataset_split)
    # normalize fields - pick first matching known field names
    def normalize_example(ex):
        if 'document' in ex and 'summary' in ex:
            return {'input': ex['document'], 'target': ex['summary']}
        if 'article' in ex and 'highlights' in ex:
            return {'input': ex['article'], 'target': ex['highlights']}
        if 'text' in ex and 'summary' in ex:
            return {'input': ex['text'], 'target': ex['summary']}
        # fallback: pick two string fields
        keys = [k for k, v in ex.items() if isinstance(v, str)]
        if len(keys) >= 2:
            return {'input': ex[keys[0]], 'target': ex[keys[1]]}
        return None

    prepared = []
    count = 0
    for ex in ds:
        if count >= dspy_max_samples:
            break
        norm = normalize_example(ex)
        if norm is None:
            continue
        prepared.append(norm)
        count += 1

    if not prepared:
        raise RuntimeError("No usable examples found in dataset split.")

    print(f"Prepared {len(prepared)} examples for DSPy evaluation.")

    # SentenceEmbedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Simple prompt template for summarization with a causal model
    PROMPT_TEMPLATE = "Summarize the following text:\n\n{input_text}\n\nSummary:"

    def gen_with_model(tokenizer, model, text: str, max_new_tokens: int = 128):
        # Works for causal models (like Granite)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=4)
        decoded = tokenizer.batch_decode(out, skip_special_tokens=True)[0]
        # If the model echoes prompt, strip the prompt prefix
        if text.strip() in decoded:
            # try to remove the prompt portion
            return decoded.replace(text.strip(), "").strip()
        return decoded.strip()

    # Generate and collect outputs
    base_outputs = []
    quant_outputs = []
    base_scores = []
    quant_scores = []
    quant_vs_base_scores = []

    for i, ex in enumerate(prepared):
        prompt = PROMPT_TEMPLATE.format(input_text=ex['input'])
        try:
            base_out = gen_with_model(base_tokenizer, base_model, prompt)
        except Exception as e:
            print(f"[WARN] base model generation failed for sample {i}: {e}")
            base_out = ""

        try:
            # quant_tokenizer/quant_model as returned by llmcompressor loader
            quant_out = gen_with_model(quant_tokenizer, quant_model, prompt)
        except Exception as e:
            print(f"[WARN] quant model generation failed for sample {i}: {e}")
            quant_out = ""

        base_outputs.append(base_out)
        quant_outputs.append(quant_out)

        # compute embedding similarities
        ref_emb = embedder.encode(ex['target'], convert_to_tensor=True)
        base_emb = embedder.encode(base_out, convert_to_tensor=True)
        quant_emb = embedder.encode(quant_out, convert_to_tensor=True)

        # cosine similarity values range [-1,1], scale to [0,1] by (x+1)/2 for readability, or clip to [0,1]
        def cos_score(a, b):
            s = util.cos_sim(a, b).item()
            # clip into [0,1] for logging convenience
            return float(max(0.0, min(1.0, (s + 1.0) / 2.0)))

        base_vs_ref = cos_score(ref_emb, base_emb)
        quant_vs_ref = cos_score(ref_emb, quant_emb)
        quant_vs_base = cos_score(base_emb, quant_emb)

        base_scores.append(base_vs_ref)
        quant_scores.append(quant_vs_ref)
        quant_vs_base_scores.append(quant_vs_base)

        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{len(prepared)} examples.")

    # Aggregate metrics
    import statistics
    metrics = {
        "base_mean_vs_ref": statistics.mean(base_scores) if base_scores else 0.0,
        "base_stdev_vs_ref": statistics.pstdev(base_scores) if base_scores else 0.0,
        "quant_mean_vs_ref": statistics.mean(quant_scores) if quant_scores else 0.0,
        "quant_stdev_vs_ref": statistics.pstdev(quant_scores) if quant_scores else 0.0,
        "quant_mean_vs_base": statistics.mean(quant_vs_base_scores) if quant_vs_base_scores else 0.0,
        "quant_stdev_vs_base": statistics.pstdev(quant_vs_base_scores) if quant_vs_base_scores else 0.0,
        "num_examples": len(prepared)
    }

    print("DSPy evaluation metrics computed:")
    print(json.dumps(metrics, indent=2))

    # Log to MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    run_name = f"dspy_eval_{os.path.basename(base_model_path)}_{os.path.basename(quantized_model_path)}"
    with mlflow.start_run(run_name=run_name):
        # Log aggregated numeric metrics
        mlflow.log_metric("base_mean_vs_ref", float(metrics["base_mean_vs_ref"]))
        mlflow.log_metric("base_stdev_vs_ref", float(metrics["base_stdev_vs_ref"]))
        mlflow.log_metric("quant_mean_vs_ref", float(metrics["quant_mean_vs_ref"]))
        mlflow.log_metric("quant_stdev_vs_ref", float(metrics["quant_stdev_vs_ref"]))
        mlflow.log_metric("quant_mean_vs_base", float(metrics["quant_mean_vs_base"]))
        mlflow.log_metric("quant_stdev_vs_base", float(metrics["quant_stdev_vs_base"]))
        mlflow.log_param("dataset", dspy_dataset_name)
        mlflow.log_param("dataset_split", dspy_dataset_split)
        mlflow.log_param("num_examples", metrics["num_examples"])
        mlflow.log_param("base_model_path", base_model_path)
        mlflow.log_param("quantized_model_path", quantized_model_path)

        # Save a sample CSV for manual inspection (first 100 examples)
        csv_path = "/tmp/dspy_comparison_sample.csv"
        with open(csv_path, "w", newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(["input_snippet", "reference", "base_output", "quant_output",
                             "base_vs_ref", "quant_vs_ref", "quant_vs_base"])
            for i, ex in enumerate(prepared[:100]):
                writer.writerow([
                    ex['input'][:300].replace("\n", " "),
                    ex['target'][:300].replace("\n", " "),
                    base_outputs[i][:300].replace("\n", " "),
                    quant_outputs[i][:300].replace("\n", " "),
                    base_scores[i],
                    quant_scores[i],
                    quant_vs_base_scores[i],
                ])
        mlflow.log_artifact(csv_path, artifact_path="dspy_comparison_samples")

    print("DSPy evaluation complete and metrics logged to MLflow.")


@dsl.pipeline(
    name="Quantization Pipeline",
    description="A pipeline for quantizing a model"
)
def quantization_pipeline(
    model_id: str="ibm-granite/granite-3.3-2b-instruct",
    output_path: str="granite-int4-pipeline",
    quantization_type: str="int4",
    eval_tasks: str="gsm8k,arc_easy",
    mlflow_tracking_uri: str="http://mlflow-server.mlflow.svc.cluster.local:8080"
):
    #Steps:
    # 1) Download model
    # 2) Quantize model
    # 3) Upload model
    # 4) Evaluate model (lm_eval)
    # 5) DSPy-based evaluation (base vs quant) and MLflow logging

    storage_class = "gp3-csi"
    secret_name = "minio-models"
    # Pipeline stage
    quantization_pvc_task = CreatePVC(
        pvc_name_suffix="-quantization",
        access_modes=["ReadWriteOnce"],
        size="30Gi",
        storage_class_name=storage_class,
    )

    download_model_task = download_model(
        model_id=model_id,
        output_path=BASE_MODEL_PATH,
    )
    download_model_task.set_caching_options(False)
    # GPU is not needed, but added to ensure the PVC is associated to a region with GPUs
    # This avoids the following tasks are not in pending status due to either having GPUs or PVC affinity
    download_model_task.set_accelerator_limit(1)
    download_model_task.set_accelerator_type("nvidia.com/gpu")
    add_toleration(download_model_task,
                   key='nvidia.com/gpu',
                   operator='Exists',
                   effect='NoSchedule')
    mount_pvc(
        task=download_model_task,
        pvc_name=quantization_pvc_task.output,
        mount_path=MOUNT_POINT,
    )

    # TODO: Only int8 and int4 supported. Add support for fp8 type
    quantize_model_task = quantize_model(
        model_path=BASE_MODEL_PATH,
        output_path=OPTIMIZED_MODEL_PATH,
        quantization_type=quantization_type,
    )
    quantize_model_task.set_caching_options(False)
    quantize_model_task.after(
        download_model_task,
    )
    quantize_model_task.set_accelerator_limit(1)
    quantize_model_task.set_accelerator_type("nvidia.com/gpu")
    add_toleration(quantize_model_task,
                   key='nvidia.com/gpu',
                   operator='Exists',
                   effect='NoSchedule')
    mount_pvc(
        task=quantize_model_task,
        pvc_name=quantization_pvc_task.output,
        mount_path=MOUNT_POINT,
    )

    upload_model_task = upload_model(
        model_path=OPTIMIZED_MODEL_PATH,
        s3_path=output_path,
    )
    upload_model_task.set_caching_options(False)
    upload_model_task.after(
        quantize_model_task,
    )
    mount_pvc(
        task=upload_model_task,
        pvc_name=quantization_pvc_task.output,
        mount_path=MOUNT_POINT,
    )
    use_secret_as_env(upload_model_task,
                      secret_name=secret_name,
                      secret_key_to_env={'AWS_ACCESS_KEY_ID': 's3_access_key',
                                         'AWS_SECRET_ACCESS_KEY': 's3_secret_access_key',
                                         'AWS_S3_ENDPOINT': 's3_host',
                                         'AWS_S3_BUCKET':'s3_bucket'})

    evaluate_model_task = evaluate_model(
        model_path=OPTIMIZED_MODEL_PATH,
        eval_tasks=eval_tasks,
    )
    evaluate_model_task.set_caching_options(False)
    evaluate_model_task.after(
        quantize_model_task,
    )
    evaluate_model_task.set_accelerator_limit(1)
    evaluate_model_task.set_accelerator_type("nvidia.com/gpu")
    add_toleration(evaluate_model_task,
                   key='nvidia.com/gpu',
                   operator='Exists',
                   effect='NoSchedule')
    mount_pvc(
        task=evaluate_model_task,
        pvc_name=quantization_pvc_task.output,
        mount_path=MOUNT_POINT,
    )

    # NEW: DSPy-based evaluation and MLflow logging, run AFTER lm_eval completes
    dspy_eval_task = dspy_evaluate_and_log(
        base_model_path=BASE_MODEL_PATH,
        quantized_model_path=OPTIMIZED_MODEL_PATH,
        mlflow_tracking_uri=mlflow_tracking_uri,
        dspy_dataset_name="xsum",
        dspy_dataset_split="validation[:512]",
        dspy_max_samples=512,
        mlflow_experiment_name="granite_quant_vs_base_dspy",
    )
    dspy_eval_task.set_caching_options(False)
    dspy_eval_task.after(evaluate_model_task)
    dspy_eval_task.set_accelerator_limit(1)
    dspy_eval_task.set_accelerator_type("nvidia.com/gpu")
    add_toleration(dspy_eval_task,
                   key='nvidia.com/gpu',
                   operator='Exists',
                   effect='NoSchedule')
    mount_pvc(
        task=dspy_eval_task,
        pvc_name=quantization_pvc_task.output,
        mount_path=MOUNT_POINT,
    )

    log_task = log_to_mlflow(
        results_json=evaluate_model_task.outputs["results_json"],
        mlflow_tracking_uri=mlflow_tracking_uri,
        model_id=model_id,
        quantization_type=quantization_type
    )
    log_task.set_caching_options(False)
    log_task.after(evaluate_model_task)

    # Keep PVC delete syncing after upload and evaluation tasks
    quantization_pvc_delete_task = DeletePVC(
        pvc_name=quantization_pvc_task.output,
    )
    quantization_pvc_delete_task.after(
        upload_model_task,
        evaluate_model_task,
        dspy_eval_task,
    )


compiler.Compiler().compile(quantization_pipeline, package_path='quantization_pipeline.yaml')
