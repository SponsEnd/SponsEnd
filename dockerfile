#Download Python from DockerHub and use it
FROM python:3.8.5

#Set the working directory in the Docker container
WORKDIR /sponsend

#Copy the dependencies file to the working directory
COPY requirements.txt .

#Install the dependencies
RUN pip install -r requirements.txt

#Copy the Flask app code to the working directory
COPY . .

#Run the container
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]