find . -type d -name "__pycache__" -exec rm -r {} +

Policy
~~~json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": "arn:aws:s3:::YOUR-BUCKET"
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::YOUR-BUCKET/*"
    }
  ]
}
~~~

~~~bash
sudo apt update
sudo apt install -y python3-pip python3-venv

mkdir -p ~/mlflow-server
cd ~/mlflow-server

python3 -m venv .venv
source .venv/bin/activate


pip install --upgrade pip
pip install mlflow boto3
mlflow --version

sudo apt install -y awscli
aws sts get-caller-identity
aws s3 ls s3://simple-mlopsp-roject

cd ~/mlflow-server
source .venv/bin/activate


mlflow server \
    --backend-store-uri sqlite:////home/ubuntu/mlflow-server/mlflow.db \
    --artifacts-destination s3://simple-mlopsp-roject/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --workers 1 \
    --allowed-hosts "*" \
    --cors-allowed-origins "*"

~~~bash