# On Productions 
## Setup
pip install -r requirements.txt
if install error - pip install ultralytics 

## Run
python app.py

## API
[POST] http://127.0.0.1:8989/growth  

[POST] http://127.0.0.1:8989/spawn

ทดสอบ เหมือนกันทั้ง 2 อัน

{
  "base64": "/9j/4AAQSkZJRgABAgAAAQABAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQo1YvbgsMlt1Vr//wO3HPWbbUNREAAAAABJRU5ErkJggg=="
}
---------------------------------

[Resp] http://127.0.0.1:8989/growth  
{
    "detections": [
        {
            "conf": 0.8665,
            "label": "normal"
        },
        {
            "conf": 0.8507,
            "label": "normal"
        }
    ],
    "image_base64": "iVBORw0KG"
}

-----------------------------------
[Resp] http://127.0.0.1:8989/spawn  

{
    "base64": "iVBORw0KGgoAAAANSUhEUgAA....",
    "white": "90.51"
}
So Far So Good
