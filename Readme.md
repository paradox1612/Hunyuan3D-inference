# Hunyuan3D Serverless API for RunPod

A **pure serverless**, high-performance API for Tencent's Hunyuan3D models (2.0 Multiview and 2.1) optimized for **RunPod Serverless** deployment. Features **pre-loaded models**, **auto-scaling from 0**, and **pay-per-inference** pricing with **Cloudflare R2** storage integration.

## Features

- **Pure Serverless Architecture**: Auto-scales 0→unlimited instances, pay only when processing
- **Pre-loaded Models**: Models baked into container (45-90 second cold starts, not minutes)
- **Dual Model Support**: Automatically switches between Hunyuan3D-2.0 (multiview) and Hunyuan3D-2.1 (single view)
- **CUDA Acceleration**: GPU-optimized inference for faster processing
- **FastAPI Backend**: RESTful API with automatic documentation
- **Redis Queue System**: In-container queuing for concurrent request handling
- **Cloudflare R2 Integration**: Automatic upload to S3-compatible storage with CDN delivery
- **Unlimited Scaling**: Handle traffic spikes from 1 to 1000+ requests simultaneously

## Model Selection Logic

- **Single Image Input**: Uses Hunyuan3D-2.1 for optimal single-view generation
- **Multiple Images Input**: Uses Hunyuan3D-2.0 with multiview capabilities

## Pure Serverless Architecture

### How It Works
```
Request → RunPod spins up container → Redis starts → Process immediately → Container shuts down
```

### **Key Benefits:**
- **$0 when idle**: No ongoing costs when not processing requests
- **Unlimited scaling**: Handle 1000+ simultaneous requests
- **Pre-loaded models**: 45-90 second cold starts (models baked in container)
- **Auto-management**: RunPod handles all infrastructure
- **Global deployment**: Deploy across multiple regions instantly

### **Container Lifecycle:**
1. **Request arrives** → RunPod detects and spins up container
2. **Container starts** (30-60s) → Models already loaded, Redis initializes
3. **Process requests** → Handle multiple jobs in queue
4. **Auto-shutdown** → Container stops after idle period (saves costs)

## RunPod Serverless Deployment

### Step 1: Build Container with Pre-loaded Models

```dockerfile
# Dockerfile optimized for RunPod Serverless
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget curl git redis-server htop \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY download_models.py .

# 🎯 PRE-LOAD MODELS (This is the key!)
RUN python download_models.py
# Models are now baked into the container (~15GB)

# Create output directory
RUN mkdir -p /app/outputs

# Health check for RunPod
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Serverless startup script
COPY start_serverless.sh .
RUN chmod +x start_serverless.sh

CMD ["./start_serverless.sh"]
```

### Step 2: Create Serverless Startup Script

```bash
# start_serverless.sh
#!/bin/bash
set -e

echo "🚀 Starting Hunyuan3D Serverless Container..."

# Start Redis in background (in-container)
echo "📊 Starting Redis..."
redis-server --daemonize yes --maxmemory 2gb --maxmemory-policy allkeys-lru

# Wait for Redis to be ready
sleep 2
redis-cli ping

# Start Celery workers in background
echo "⚡ Starting Celery workers..."
celery -A src.worker worker --loglevel=info --concurrency=2 --detach

# Start FastAPI server
echo "🌐 Starting API server..."
exec uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Step 3: Deploy to RunPod Serverless

```bash
# Build and push container (will be ~20-25GB with models)
docker build -t your-username/hunyuan3d-serverless:latest .
docker push your-username/hunyuan3d-serverless:latest
```

**RunPod Console Settings:**
```yaml
Container Image: your-username/hunyuan3d-serverless:latest
GPU: A100 (40GB) or A100 (80GB) recommended
RAM: 32GB+
Container Disk: 50GB+ (for model storage)
Ports: 8000

Environment Variables:
  R2_ENDPOINT_URL: https://your-account-id.r2.cloudflarestorage.com
  R2_ACCESS_KEY_ID: your-r2-access-key
  R2_SECRET_ACCESS_KEY: your-r2-secret-key
  R2_BUCKET_NAME: hunyuan3d-outputs
  R2_PUBLIC_URL: https://pub-your-bucket-id.r2.dev

Scaling:
  Min Workers: 0
  Max Workers: 10 (or higher based on expected load)
  Scale Down Delay: 5 minutes
  Request Timeout: 300 seconds
```

### Manual Installation

1. **Install dependencies**
```bash
pip install fastapi uvicorn celery[redis] redis python-multipart Pillow torch torchvision transformers diffusers accelerate xformers boto3
```

2. **Download models** (automatic on first run)
```bash
python download_models.py
```

3. **Start Redis server**
```bash
redis-server
```

4. **Start Celery workers** (in separate terminals)
```bash
# Worker 1
celery -A src.worker worker --loglevel=info --concurrency=1

# Worker 2 (if you have multiple GPUs)
CUDA_VISIBLE_DEVICES=1 celery -A src.worker worker --loglevel=info --concurrency=1
```

5. **Run the API**
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

6. **Optional: Start Flower monitoring**
```bash
celery -A src.worker flower --port=5555
```

## API Usage

### Create Prediction

Replace your existing Synexa API call with:

```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tencent/hunyuan3d-2",
    "input": {
      "seed": 1234,
      "image": "https://example.com/image.png",
      "steps": 5,
      "caption": "",
      "shape_only": true,
      "guidance_scale": 5.5,
      "check_box_rembg": true,
      "octree_resolution": "256",
      "multiple_views": []
    }
  }' \
  http://localhost:8000/v1/predictions
```

**Response:**
```json
{
  "id": "7c5b46ca-9f83-4508-98de-f48d2b0f2c71",
  "model": "tencent/hunyuan3d-2.1",
  "version": null,
  "input": { ... },
  "logs": null,
  "output": null,
  "error": null,
  "status": "starting",
  "created_at": "2025-08-07T03:36:44.326000+00:00",
  "started_at": null,
  "completed_at": null,
  "metrics": null
}
```

### Check Prediction Status

```bash
curl -s -X GET \
  -H "Content-Type: application/json" \
  http://localhost:8000/v1/predictions/<prediction-id>
```

**Response (Processing):**
```json
{
  "id": "7c5b46ca-9f83-4508-98de-f48d2b0f2c71",
  "status": "processing",
  "progress": 0.3,
  "logs": "Loading model...",
  ...
}
```

**Response (Completed):**
```json
{
  "id": "7c5b46ca-9f83-4508-98de-f48d2b0f2c71",
  "model": "tencent/hunyuan3d-2.1",
  "version": null,
  "input": {
    "seed": 1234,
    "image": "https://example.com/image.png",
    "steps": 5,
    "caption": "",
    "shape_only": true,
    "guidance_scale": 5.5,
    "multiple_views": [],
    "check_box_rembg": true,
    "octree_resolution": "256"
  },
  "logs": null,
  "output": [
    "https://your-r2-bucket.your-domain.com/outputs/7c5b46ca-9f83-4508-98de-f48d2b0f2c71_white_mesh.glb"
  ],
  "error": null,
  "status": "succeeded",
  "created_at": "2025-08-07T03:37:02.593000+00:00",
  "started_at": "2025-08-07T03:38:51.850000+00:00",
  "completed_at": "2025-08-07T03:38:59.167000+00:00",
  "metrics": {
    "predict_time": 7.317,
    "upload_time": 1.2
  }
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/predictions` | POST | Create new prediction job |
| `/v1/predictions/{id}` | GET | Get prediction status and results |
| `/health` | GET | Health check endpoint |
| `/docs` | GET | Interactive API documentation |

## Cloudflare R2 Setup

### 1. Create R2 Bucket

1. **Login to Cloudflare Dashboard**
2. **Go to R2 Object Storage**
3. **Create a new bucket**: `hunyuan3d-outputs`
4. **Configure bucket settings**:
   - Enable public access for generated files
   - Set up custom domain (optional): `files.yourdomain.com`

### 2. Get R2 Credentials

1. **Go to R2 → Manage R2 API tokens**
2. **Create API token** with permissions:
   - Object Read & Write
   - Bucket access: `hunyuan3d-outputs`
3. **Note down**:
   - Access Key ID
   - Secret Access Key
   - Account ID

### 3. Configure R2 Environment

```env
# Add to your .env file
R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key-here
R2_SECRET_ACCESS_KEY=your-secret-key-here
R2_BUCKET_NAME=hunyuan3d-outputs
R2_PUBLIC_URL=https://pub-your-bucket-id.r2.dev
# OR with custom domain:
# R2_PUBLIC_URL=https://files.yourdomain.com
```

### 4. File Upload Workflow

```python
# Automatic workflow after inference:
def process_inference_job(job_data):
    # 1. Generate 3D model
    glb_file = run_hunyuan3d_inference(job_data)
    
    # 2. Save locally first (temporary)
    local_path = f"/tmp/{job_id}_white_mesh.glb"
    
    # 3. Upload to R2 
    r2_url = upload_to_r2(local_path, f"outputs/{job_id}_white_mesh.glb")
    
    # 4. Cleanup local file
    os.remove(local_path)
    
    # 5. Return R2 URL in response
    return {
        "output": [r2_url],
        "status": "succeeded"
    }
```

## Output Format & Response Structure

The API returns the exact same response structure as Synexa API for seamless migration, but with R2 URLs instead of local files:

### Success Response Structure
```json
{
  "id": "prediction-uuid",
  "model": "tencent/hunyuan3d-2" or "tencent/hunyuan3d-2.1", 
  "version": null,
  "input": { /* original input parameters */ },
  "logs": null,
  "output": [
    "http://localhost:8000/outputs/{id}_white_mesh.glb"
  ],
  "error": null,
  "status": "succeeded",
  "created_at": "2025-08-07T03:37:02.593000+00:00",
  "started_at": "2025-08-07T03:38:51.850000+00:00", 
  "completed_at": "2025-08-07T03:38:59.167000+00:00",
  "metrics": {
    "predict_time": 7.317
  }
}
```

### Status Values
- `"starting"`: Job has been queued
- `"processing"`: Model is running inference  
- `"succeeded"`: Job completed successfully
- `"failed"`: Job encountered an error

### Output Files & Storage
- **Format**: GLB (GL Transmission Format) - same as Synexa
- **Naming**: `{prediction_id}_white_mesh.glb`
- **Storage**: Cloudflare R2 with global CDN delivery  
- **Access**: Direct HTTPS download URLs with high availability
- **Persistence**: Files stored permanently in R2 bucket
- **Performance**: CDN-cached for fast global access

### R2 Integration Benefits
- **Global CDN**: Fast file delivery worldwide via Cloudflare
- **Cost Efficient**: R2 pricing significantly lower than AWS S3
- **High Availability**: 99.9% uptime with redundant storage
- **No Egress Fees**: Free bandwidth for file downloads
- **Custom Domains**: Use your own domain for professional URLs

## Input Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `seed` | integer | No | 1234 | Random seed for reproducibility |
| `image` | string/array | Yes | - | Single image URL or array of image URLs |
| `steps` | integer | No | 5 | Number of inference steps |
| `caption` | string | No | "" | Text description (optional) |
| `shape_only` | boolean | No | true | Generate geometry only |
| `guidance_scale` | float | No | 5.5 | Guidance strength |
| `check_box_rembg` | boolean | No | true | Remove background |
| `octree_resolution` | string | No | "256" | Output resolution ("128", "256", "512") |
| `multiple_views` | array | No | [] | Additional view angles |

## Project Structure

```
hunyuan3d-api/
├── src/
│   ├── main.py              # FastAPI application
│   ├── worker.py            # Celery worker for background jobs
│   ├── models/              # Model loading and inference
│   │   ├── hunyuan3d_21.py  # Hunyuan3D-2.1 implementation
│   │   └── hunyuan3d_20_mv.py # Hunyuan3D-2.0 multiview
│   ├── api/                 # API routes and schemas
│   │   ├── routes.py
│   │   └── schemas.py
│   ├── core/                # Core functionality
│   │   ├── queue_manager.py # Redis queue management
│   │   ├── job_processor.py # Job processing logic
│   │   ├── gpu_manager.py   # CUDA memory management
│   │   └── utils.py         # Utilities
│   ├── storage/             # File storage handling
│   │   ├── local_storage.py # Local file operations
│   │   └── r2_uploader.py   # Cloudflare R2 integration
│   └── tasks/               # Celery tasks
│       └── inference_tasks.py
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── worker.Dockerfile    # Separate worker container
├── requirements.txt
├── download_models.py
└── README.md
```

## Environment Variables

Create a `.env` file:

```env
## RunPod Configuration

### Environment Variables for RunPod
```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Redis Queue Configuration (High-throughput settings)
REDIS_URL=redis://localhost:6379/0
REDIS_MAX_CONNECTIONS=100
MAX_CONCURRENT_JOBS=10
QUEUE_NAME=hunyuan3d_jobs
JOB_TIMEOUT=600
RESULT_TTL=7200
QUEUE_HIGH_WATERMARK=1000
QUEUE_LOW_WATERMARK=100

# Worker Configuration (RunPod optimized)
CELERY_WORKERS=8
WORKER_CONCURRENCY=1
WORKER_PREFETCH_MULTIPLIER=1
CELERYD_MAX_TASKS_PER_CHILD=100
CELERY_TASK_ROUTES={'src.tasks.inference_tasks.process_hunyuan3d': {'queue': 'gpu_queue'}}

# GPU Configuration (RunPod auto-detected)
CUDA_VISIBLE_DEVICES=auto
GPU_MEMORY_LIMIT=auto
ENABLE_MIXED_PRECISION=true
MODEL_HALF_PRECISION=true

# RunPod Specific
RUNPOD_ENDPOINT_ID=${RUNPOD_ENDPOINT_ID}
RUNPOD_POD_ID=${RUNPOD_POD_ID}
RUNPOD_VOLUME=/runpod-volume

# Cloudflare R2 Configuration (S3-Compatible Storage)
R2_ENDPOINT_URL=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-r2-access-key
R2_SECRET_ACCESS_KEY=your-r2-secret-key
R2_BUCKET_NAME=hunyuan3d-outputs
R2_REGION=auto
R2_PUBLIC_URL=https://your-r2-bucket.your-domain.com

# Upload Configuration
UPLOAD_TO_R2=true
LOCAL_FILE_CLEANUP=true
CLEANUP_AFTER_UPLOAD=true
UPLOAD_TIMEOUT=30
MAX_UPLOAD_RETRIES=3

# Logging (Production settings)
LOG_LEVEL=INFO
CELERY_LOG_LEVEL=WARNING
ENABLE_STRUCTURED_LOGGING=true

# Storage (RunPod volume + R2)
OUTPUT_DIR=/runpod-volume/outputs
MODEL_CACHE_DIR=/runpod-volume/models
CLEANUP_FILES_AFTER=1h
MAX_OUTPUT_FILES=1000

# Performance (High-throughput)
UVICORN_WORKERS=4
KEEPALIVE_TIMEOUT=30
MAX_REQUEST_SIZE=50MB
REQUEST_TIMEOUT=300
```
```

## Docker Configuration

### Serverless-Optimized Dockerfile
```dockerfile
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Install system dependencies (minimal for serverless)
RUN apt-get update && apt-get install -y \
    wget curl git redis-server \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy application code
COPY src/ ./src/
COPY download_models.py .
COPY start_serverless.sh .

# 🎯 CRITICAL: Pre-download models (15-20GB)
RUN python download_models.py && \
    rm download_models.py && \
    find /root/.cache -type f -delete

# Create directories and set permissions
RUN mkdir -p /app/outputs /tmp/redis && \
    chmod +x start_serverless.sh && \
    chmod -R 755 /app

# Health check for RunPod monitoring
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Serverless entry point
CMD ["./start_serverless.sh"]
```

### Docker Compose with Queue System
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  hunyuan3d-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - REDIS_URL=redis://redis:6379/0
      - MAX_CONCURRENT_JOBS=2
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Worker for processing jobs
  hunyuan3d-worker:
    build: .
    command: celery -A src.worker worker --loglevel=info --concurrency=1
    volumes:
      - ./outputs:/app/outputs
      - ./models:/app/models
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    # Scale workers based on your GPU capacity
    scale: 2

  # Optional: Celery monitoring
  flower:
    build: .
    command: celery -A src.worker flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis

volumes:
  redis_data:
```

## Monitoring & Troubleshooting

### **Serverless Monitoring:**
- **RunPod Dashboard**: Real-time container scaling and costs
- **Container Logs**: Access via RunPod console for debugging  
- **Request Metrics**: Built-in latency and success rate tracking
- **Cost Tracking**: Per-second billing visibility

### **Common Issues:**

**Cold Start Too Slow:**
```bash
# Optimize container image size
# Remove unnecessary dependencies
# Use multi-stage builds
docker build --squash -t username/hunyuan3d:optimized .
```

**R2 Upload Failures:**
```bash
# Test R2 connectivity from container
curl -I https://pub-your-bucket-id.r2.dev/
aws configure set aws_access_key_id $R2_ACCESS_KEY_ID
aws configure set aws_secret_access_key $R2_SECRET_ACCESS_KEY  
aws s3 ls s3://your-bucket --endpoint-url=$R2_ENDPOINT_URL
```

**Container Memory Issues:**
```bash
# Monitor GPU memory in container
nvidia-smi
# Reduce model precision or octree resolution
octree_resolution="128"  # Instead of "256"
```

**High Costs:**
```bash
# Monitor scaling behavior
# Optimize container shutdown timing
CONTAINER_IDLE_TIMEOUT=180  # Shutdown after 3 minutes
```

### **Performance Optimization Tips:**

- **Container Image**: Keep under 25GB for faster cold starts
- **Memory Management**: Use `torch.cuda.empty_cache()` after each job
- **R2 Uploads**: Use multipart uploads for files >100MB
- **Redis Config**: Limit memory usage to prevent OOM
- **Monitoring**: Track costs daily to avoid surprises



## Production Deployment Guide

### **Step-by-Step Serverless Deployment:**

#### 1. Prepare Container Image
```bash
# Clone and build
git clone <your-repo>
cd hunyuan3d-serverless-api

# Build with pre-loaded models (this will take 20-30 minutes)
docker build -t username/hunyuan3d-serverless:v1.0 .

# Push to registry
docker push username/hunyuan3d-serverless:v1.0
```

#### 2. Set Up Cloudflare R2
```bash
# Create R2 bucket
# 1. Login to Cloudflare Dashboard
# 2. Navigate to R2 Object Storage  
# 3. Create bucket: "hunyuan3d-outputs"
# 4. Enable public access
# 5. Optional: Set custom domain

# Get R2 credentials
# Navigate to R2 → Manage R2 API tokens
# Create token with Object Read & Write permissions
```

#### 3. Deploy on RunPod Serverless
```yaml
# RunPod Console Configuration:

Template Name: Hunyuan3D Serverless API
Container Image: username/hunyuan3d-serverless:v1.0
Container Registry Credentials: [Your Docker Hub credentials]

GPU Configuration:
  GPU Type: RTX A5000, A100 40GB, or A100 80GB
  vCPU: 8-16 cores  
  RAM: 32-64GB
  Container Disk: 50GB
  Volume Disk: Not needed (models in container)

Network:
  HTTP Ports: 8000
  TCP Ports: None

Environment Variables:
  R2_ENDPOINT_URL: https://[account-id].r2.cloudflarestorage.com
  R2_ACCESS_KEY_ID: [your-access-key]
  R2_SECRET_ACCESS_KEY: [your-secret-key]  
  R2_BUCKET_NAME: hunyuan3d-outputs
  R2_PUBLIC_URL: https://pub-[bucket-id].r2.dev

Serverless Configuration:
  Min Workers: 0
  Max Workers: 20 (adjust based on expected load)
  Idle Timeout: 5 minutes
  Max Wait Time: 20 seconds
  Flashboot: Enabled (for faster cold starts)
```

#### 4. Test Deployment
```bash
# Get your RunPod endpoint URL
ENDPOINT_URL="https://[your-runpod-id]-8000.proxy.runpod.net"

# Test health check
curl $ENDPOINT_URL/health

# Test inference (will trigger cold start)
curl -X POST $ENDPOINT_URL/v1/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tencent/hunyuan3d-2",
    "input": {
      "seed": 12345,
      "image": "https://picsum.photos/512",
      "steps": 5,
      "guidance_scale": 5.5,
      "octree_resolution": "256"
    }
  }'
```

### **Production Best Practices:**

#### Container Optimization
```dockerfile
# Multi-stage build for smaller image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04 as builder

# Download models in builder stage
COPY download_models.py .
RUN python download_models.py

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Copy pre-downloaded models
COPY --from=builder /root/.cache /root/.cache
COPY src/ ./src/

# Optimize for production
RUN pip install --no-cache-dir fastapi uvicorn celery redis boto3 && \
    rm -rf /var/lib/apt/lists/* && \
    pip cache purge
```

#### Error Handling & Logging
```python
# Enhanced error handling for production
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "id": str(uuid.uuid4()),
            "status": "failed"
        }
    )

# Structured logging
import structlog
logger = structlog.get_logger()
```

#### Cost Monitoring
```python
# Track costs and usage
def track_usage(job_id: str, processing_time: float, gpu_type: str):
    cost_per_hour = {
        "A5000": 0.45,
        "A100-40GB": 1.50, 
        "A100-80GB": 2.50
    }
    
    job_cost = (processing_time / 3600) * cost_per_hour.get(gpu_type, 1.50)
    
    # Log to monitoring system
    logger.info("job_completed", 
                job_id=job_id, 
                processing_time=processing_time,
                estimated_cost=job_cost)
```




#### Key Metrics to Track
```python
# Essential production metrics
METRICS_TO_MONITOR = {
    "request_rate": "requests per minute",
    "cold_start_time": "container startup latency",
    "processing_time": "inference + upload time", 
    "error_rate": "failed requests percentage",
    "cost_per_job": "estimated cost per inference",
    "r2_upload_success": "file upload success rate",
    "queue_depth": "pending jobs in queue"
}
```

#### Alerting Setup
```yaml
# Example alert conditions
Alerts:
  High Error Rate:
    Condition: error_rate > 5%
    Action: Slack notification
    
  Long Cold Starts:
    Condition: cold_start_time > 120 seconds
    Action: Email alert + container optimization needed
    
  High Costs:
    Condition: daily_cost > $500
    Action: Cost alert + scaling review
    
  R2 Upload Failures:
    Condition: r2_upload_success < 95%
    Action: Check R2 connectivity
```


## Summary

You now have a **production-ready, pure serverless** Hunyuan3D API that:

✅ **Scales automatically** from 0 to unlimited based on demand  
✅ **Costs $0 when idle** - only pay for actual inference time  
✅ **Handles traffic spikes** with instant container scaling  
✅ **Delivers files globally** via Cloudflare R2 CDN  
✅ **Maintains API compatibility** with existing Synexa workflows  
✅ **Provides full control** over models, performance, and costs  

**Expected Performance**: 45-90 second cold starts, then 7-12 second inference times with unlimited concurrent capacity and global file delivery.

This setup can easily handle anything from hobby projects to enterprise workloads while maintaining cost efficiency and professional reliability! 🚀marks



**Monitor Queue Performance:**
```python
# monitor_queue.py
import redis
import time
import requests

def monitor_performance():
    redis_client = redis.Redis(host='your-redis-host', port=6379, db=0)
    
    while True:
        queue_size = redis_client.llen('hunyuan3d_jobs')
        active_workers = get_active_workers()
        
        print(f"Queue Size: {queue_size}")
        print(f"Active Workers: {active_workers}")
        print(f"Est. Wait Time: {queue_size / active_workers * 7.3:.1f} seconds")
        
        time.sleep(10)
```



**Flower Dashboard Metrics:**
- Access at `https://your-runpod-id-5555.proxy.runpod.net`
- Real-time worker status
- Task success/failure rates
- Processing time distributions
