ABSOLUTE_PATH := $(shell pwd)
BASE_IMAGE_NAME := predictor-system
IMAGE_VERSION := 0.0.1

DOCKERFILE := Dockerfile

build-dev:
	pip install -r requirements.txt

build-env:
	docker build --no-cache -t ${BASE_IMAGE_NAME}:${IMAGE_VERSION} -f ${DOCKERFILE} .

load-train:
	mlflow run . --env-manager=local

solar-train:
	mlflow run . -P preprocess_data=solar --env-manager=local

run-ui:
	mlflow ui