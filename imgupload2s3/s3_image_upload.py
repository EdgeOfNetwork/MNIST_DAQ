import yaml
import boto3
from botocore.exceptions import ClientError

""" Ready for S3 session """


def load_config():
    with open('config.yml', 'r') as config_file:
        config = yaml.load(config_file)
        return config


def initiate_session(config, client):
    session = boto3.Session(
        aws_access_key_id=config['aws_access_key_id'],
        aws_secret_access_key=config['aws_secret_access_key'],
        region_name='ap-northeast-2'
    )
    print(session)
    client = session.resource(client)
    return client


def upload_file(client, fileobj, bucket, key):
    with open(fileobj, 'rb') as data:
        try:
            client.Bucket(bucket).put_object(
                Body=data,
                Bucket=bucket,
                Key=key,
                ContentType='image/jpeg'
            )
            print('success')
            return 'success'

        except ClientError as e:
            print('error: %s') % e
            return 'error'

