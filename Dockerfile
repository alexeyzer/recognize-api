FROM python:3.9

# copy the content of the local src directory to the working directory
COPY . .

# install dependencies
RUN pip install -r requirements.txt

EXPOSE 8082

# command to run on container start
CMD [ "python", "./main.py" ]