import boto3

def read_file_from_s3(bucket : str, key: str) -> str:
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    return response['Body'].read().decode('utf-8')


if __name__ == "__main__":
    print(read_file_from_s3("risk-lens","regimes/main_regime_model.json"))