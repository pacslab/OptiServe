FROM public.ecr.aws/lambda/python:3.12

COPY models.py app.py requirements.txt ./

# temporarily install gcc so that scikit-learn can be built
RUN dnf install gcc-c++ -y

RUN python3.12 -m pip install -r requirements.txt -t .

CMD ["app.lambda_handler_resnet"]
