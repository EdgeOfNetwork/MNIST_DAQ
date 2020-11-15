import imgupload2s3.s3_image_upload
import psycopg2
# from pymongo import MongoClient
import pymongo


def connect_mongo():
    connection = pymongo.MongoClient('localhost', 27017)


if __name__ == '__main__':
    connect_mongo()

    # config = load_config()
    # client = initiate_session(config, 's3')
    #
    # """ target """
    # file_obj = './static/great_success.jpg'
    # bucket = config['upload_bucket']
    # key = 'testimg.jpg'
    #
    # upload_file(client, file_obj, bucket, key)
