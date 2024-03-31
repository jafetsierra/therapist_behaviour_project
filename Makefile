IMAGE_NAME ?= therapist_behaviour
IMAGE_TAG ?= latest

build_amd64:
	docker build --platform linux/amd64 \
	-t $(IMAGE_NAME):$(IMAGE_TAG) \
	-f Dockerfile .

build_arm64:
	docker build --platform linux/arm64 \
	-t $(IMAGE_NAME):$(IMAGE_TAG) \
	-f Dockerfile .