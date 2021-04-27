# OpenCV Document Scanner

A demo API for document detection and extraction using:
* [OpenCV](https://docs.opencv.org/master/)
* [WebSocket](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)
* [FastAPI](https://fastapi.tiangolo.com/advanced/websockets/)
* [Docker](https://docs.docker.com/reference/)

A [simple demo IPython notebook can be found here](demo.ipynb).

# Usage

## TL;DR

### Deployment

* Launch (and teardown) the docker containers for the FastAPI server using the following commands:
```shell script
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker-compose down --remove-orphans
```

* Request: The client shall send a JSON request containing these fields:
    
    * `image`: A string containing image content encoded in Base64.
    * `doc_type`: An integer (optional, defaults to 0).
        Indicates document type. This is ignored for now.

* Response: The client shall receive a JSON response containing these fields:
    
    * `det_success`: A boolean. Indicates whether document detection succeeded or not.
    * `doc`: Extracted document image encoded in Base64.
    * `doc_points`: A list of floats. The corner points of the detected document.
    * `doc_vis`: Visualisation of the document image after some pre-processing, encoded in Base64.

* Environment variables can be set in the `.env` file in root directory.

### Debug / Visualisation

After the server is launched, one can navigate to `http://localhost:5000/` for a test page.

### Development

```shell script
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
docker-compose ps       # List containers
docker-compose exec scanner bash
docker-compose down --remove-orphans
```
```shell script
docker build -t scanner/python:3.7.10 .
docker run -it --ipc=host -v %cd%:/master/scanner -p 5000:5000 --rm scanner/python:3.7.10
docker run -it --ipc=host -v $(pwd):/master/scanner -p 5000:5000 --rm scanner/python:3.7.10
```

## Explanation
```
User <-- WebSocket --> FastAPI <---> Document Detector
```

1. Client will communicate with our FastAPI framework via WebSocket protocol.
The client shall send an image containing the document to be extracted (the query image).
This query image will then be processed by the detectors.

2. The document detector can operate in 1 of 2 modes: `simple` and `features`. 
This can be controlled by setting the `DET_SIFT_FEATURE` environment variable.

    * Simple (`DET_SIFT_FEATURE = False`, default): 
    Otsu thresholding is performed on hue image after some pre-processing. 
    The document corner points are then estimated via a contour operation.
    _This mode is faster, but relies heavily having a background that is clean (one colour) 
    and with large contrast (different colour than the document)._
    
    * Features (`DET_SIFT_FEATURE = True`): 
    Local features (default = SIFT) are extracted from a reference document image and the query image.
    These image features / keypoints are then matched using either a brute-force matcher (default) or a FLANN-based matcher.
    The matched keypoints are then used to estimate a homography matrix using either LMEDS (default) or RANSAC.
    The homography matrix is then used to compute the document corner points.
    _This mode is slower, but should be more flexible and less reliant on having a clean background._

    After the document corner points are obtained, perspective transform is performed to extract the document 
    from the query image.


# Limitations / Known Issues

## FastAPI / Uvicorn

1. Websocket will disconnect when uploading a base64 file with size > 1MB
    * Alternatively, use Hypercorn which has a [default message size limit of 16MB](
        https://pgjones.gitlab.io/hypercorn/discussion/dos_mitigations.html#large-websocket-message)
    * Links:
        * https://github.com/tiangolo/fastapi/issues/2071
        * https://github.com/encode/uvicorn/issues/432
        * https://github.com/encode/uvicorn/pull/538
        * https://github.com/Opentrons/opentrons/issues/6159


# References

## FastAPI

* Deployment
    * https://fastapi.tiangolo.com/deployment/manually/
    * https://www.uvicorn.org/deployment/#gunicorn
    * https://pgjones.gitlab.io/hypercorn/how_to_guides/configuring.html#configuration-options

## Installing `docker-compose` on Linux

[Guide](https://docs.docker.com/compose/install/)

```shell script
# Download the current stable release of Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/1.28.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
# Apply executable permissions to the binary
sudo chmod +x /usr/local/bin/docker-compose
# Create a symbolic link
sudo ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
# Check version
docker-compose --version
# Command completion (optional)
sudo curl -L https://raw.githubusercontent.com/docker/compose/1.28.5/contrib/completion/bash/docker-compose -o /etc/bash_completion.d/docker-compose
```
