import boto3
import json

# Initialize a SageMaker runtime client
# Configure AWS credentials
# Option 1: Set environment variables:
#   export AWS_ACCESS_KEY_ID='your_access_key'
#   export AWS_SECRET_ACCESS_KEY='your_secret_key'

# Option 2: Add credentials directly (not recommended for production):
client = boto3.client(
    "sagemaker-runtime",
    region_name="ap-southeast-1",
    aws_access_key_id="your_access_key",
    aws_secret_access_key="your_secret_key",
)

response = client.invoke_endpoint(
    EndpointName="jumpstart-dft-llama-3-2-1b-instruct-20250208-085355", # Remame this to your endpoint name
    Body=json.dumps(
        {
            "inputs": "Where do I learn about neural networks?",
            "parameters": {
                "max_new_tokens": 128,
                "top_p": 0.9,
                "temperature": 0.6,
                "details": False,
            },
        }
    ).encode(),
    ContentType="application/json",
)

print(response["Body"].read())
