FROM python:3.9.4
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8000"]
#CMD ["python", "-m", "flask", "--app", "app_yseult", "run", "--host=0.0.0.0", "--port=8000"]


#Put in the readme

#To create the image: docker build . -t puzzle_image

#docker run -p 8000:8000 puzzle_image
#If http://0.0.0.0:8000/ doesn't work, try using http://127.0.0.1:8000/ or http://localhost:8000/