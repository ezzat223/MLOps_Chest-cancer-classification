# End-to-End-Chest-Cancer-Classification-using-MLflow-DVC


##### cmd
- mlflow ui


### DVC cmd
1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC
MLflow
 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 
 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



# AWS-CICD-Deployment-with-Github-Actions
#Description: About the deployment
	1. Build docker image of the source code.
	2. Push your docker image to ECR.
	3. Launch Your EC2.
	4. Pull Your image from ECR in EC2.
	5. Lauch your docker image in EC2.

## 1. Login to AWS console.
## 2. Create IAM user for deployment:
	#Policy:
	1. AmazonEC2ContainerRegistryFullAccess
	2. AmazonEC2FullAccess

	- Create Access key for CLI.
		- Add the Access key and secret access key as secrets in GitHub.

## 3. Create ECR repo to store/save docker image
    - name: kidney
	- Save its URI: 471112551571.dkr.ecr.us-east-1.amazonaws.com/kidney

## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	#optinal
	sudo apt-get update -y
	sudo apt-get upgrade
	
	#required
	curl -fsSL https://get.docker.com -o get-docker.sh
	sudo sh get-docker.sh
	sudo usermod -aG docker ubuntu
	newgrp docker

	#test installaion
	docker --version

# 6. Configure EC2 as self-hosted runner for GitHub:
    In repo's setting > actions > runner > new self hosted runner> choose os > then run command one by one on the EC2 Instance.

# 7. Setup github secrets:
    AWS_ACCESS_KEY_ID=
    AWS_SECRET_ACCESS_KEY=
    AWS_REGION = us-east-1
    AWS_ECR_LOGIN_URI =  471112551571.dkr.ecr.us-east-1.amazonaws.com  (remove the repo name)
    ECR_REPOSITORY_NAME = kidney