#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
import boto3
import re
import sagemaker


role = sagemaker.get_execution_role()
region = boto3.Session().region_name

# S3 bucket for training data.
data_bucket = sagemaker.Session().default_bucket()
data_prefix = "sagemaker/lin_reg_in"

# S3 bucket for saving code and model artifacts.
output_bucket = sagemaker.Session().default_bucket()
output_prefix = "sagemaker/lin_reg_out"


# In[21]:


get_ipython().run_cell_magic('time', '', '\nimport io\nimport boto3\nimport random\n\n# Loading the dataset \ndata_file = "data.txt"\n\n# spliting the downloaded data into train and test files\n\nTRAIN_FILE = "train.txt"\nTEST_FILE = "test.txt"\nTRAIN_PERCENT = 70\nTEST_PERCENT = 30\n\ndef data_split(data_file, TRAIN_FILE, TEST_FILE, TRAIN_PERCENT, TEST_PERCENT):\n    \n    data = [l for l in open(data_file, "r")]\n    train_file = open(TRAIN_FILE, "w")\n    tests_file = open(TEST_FILE, "w")\n\n    num_of_data = len(data)\n    num_train = int((TRAIN_PERCENT / 100.0) * num_of_data)\n    num_tests = int((TEST_PERCENT / 100.0) * num_of_data)\n\n    data_fractions = [num_train, num_tests]\n    split_data = [[], []]\n\n    rand_data_ind = 0\n\n    for split_ind, fraction in enumerate(data_fractions):\n        for i in range(fraction):\n            rand_data_ind = random.randint(0, len(data) - 1)\n            split_data[split_ind].append(data[rand_data_ind])\n            data.pop(rand_data_ind)\n\n    for l in split_data[0]:\n        train_file.write(l)\n\n    for l in split_data[1]:\n        tests_file.write(l)\n\n    train_file.close()\n    tests_file.close()\n    \ndata_split(data_file, TRAIN_FILE, TEST_FILE, TRAIN_PERCENT, TEST_PERCENT)\n\n\n# S3 bucket to store training data.\n\nbucket = sagemaker.Session().default_bucket()\nprefix = "sagemaker/lin_reg"\n\ndef write_to_s3(fobj, bucket, key):\n    return (boto3.Session(region_name=region).resource("s3").Bucket(bucket).Object(key).upload_fileobj(fobj))\n\n\ndef upload_to_s3(bucket, prefix, channel, filename):\n    fobj = open(filename, "rb")\n    key = f"{prefix}/{channel}/{filename}"\n    url = f"s3://{bucket}/{key}"\n    print(f"Writing to {url}")\n    write_to_s3(fobj, bucket, key)\n    \n# uploading the files to the S3 bucket\n\nupload_to_s3(bucket, prefix, "train", TRAIN_FILE)\nupload_to_s3(bucket, prefix, "test", TEST_FILE)')


# In[22]:


# creating the inputs for the fit() function with the training location

s3_train_data = f"s3://{data_bucket}/{data_prefix}/train"
print(f"training files will be taken from: {s3_train_data}")
output_location = f"s3://{output_bucket}/{output_prefix}/output"
print(f"training artifacts output location: {output_location}")

# generating the session.s3_input() format for fit() accepted by the sdk

train_data = sagemaker.inputs.TrainingInput(
    s3_train_data,
    distribution="FullyReplicated",
    content_type="text/csv",
    s3_data_type="S3Prefix",
    record_wrapping=None,
    compression=None,)


# In[23]:


# getting the linear learner image according to the region

from sagemaker.image_uris import retrieve

container = retrieve("linear-learner", boto3.Session().region_name, version="1")
print(container)


# In[34]:


get_ipython().run_cell_magic('time', '', '\nfrom time import gmtime, strftime\n\nsess = sagemaker.Session()\n\njob_name = "linear-regressor" + strftime("%H-%M-%S", gmtime())\nprint("Training job", job_name)\n\nlinear = sagemaker.estimator.Estimator(\n    container,\n    role,\n    input_mode="File",\n    instance_count=1,\n    instance_type="ml.c4.xlarge",\n    output_path=output_location,\n    sagemaker_session=sess,)\n\nlinear.set_hyperparameters(\n    feature_dim=11,\n    epochs=800,\n    loss="absolute_loss",\n    predictor_type="regressor",\n    normalize_data=True,\n    optimizer="adam",\n    mini_batch_size=80,\n    beta_1=0.5\n    beta_2=0.99\n    learning_rate=0.0001,)\n\nlinear.fit(inputs={"train": train_data})')


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# creating the endpoint out of the trained model\n\nlinear_predictor = linear.deploy(initial_instance_count=1, instance_type="ml.c4.large")\nprint(f"\\ncreated endpoint: {linear_predictor.endpoint_name}")')


# In[ ]:


# configure the predictor to accept to serialize csv input and parse the reposne as json

from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

linear_predictor.serializer = CSVSerializer()
linear_predictor.deserializer = JSONDeserializer()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nimport json\nfrom itertools import islice\nimport math\nimport struct\nimport boto3\nimport random\n\n# downloading the test file from data_bucket\n\nTEST_FILE = "test.txt"\ns3 = boto3.client("s3")\ns3.download_file(data_bucket, f"{data_prefix}/test/{TEST_FILE}", TEST_FILE)\n\n# getting testing sample from the test file\n\ntest_data = [l for l in open(TEST_FILE, "r")]\nsample = random.choice(test_data).split(",")\nREAL = sample[0]\n\n# removing the label from the sample\n\nFEATURES = sample[1:]  \nFEATURES = ",".join(map(str, FEATURE))\n\n# Invoke the predictor and analyse the results\n\nresult = linear_predictor.predict(FEATURES)\n\n# extracting the prediction value\n\nresult = round(float(result["predictions"][0]["score"]), 2)')


# In[ ]:


sagemaker.Session().delete_endpoint(linear_predictor.endpoint_name)
print(f"deleted {linear_predictor.endpoint_name} successfully!")

