from fastapi import FastAPI, Request, Depends
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import models, crud, database, utils, schemas
from database import engine, SessionLocal
from starlette.concurrency import run_in_threadpool
from jinja2 import Environment, FileSystemLoader

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="C:/Users/balde/OneDrive/Bureau/DA_DS/FASTAPI/sentiment-analysis-app/templates")




def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/generate")
async def generate_content(payload: schemas.GeneratePayload, db: Session = Depends(get_db)):
    generate_text = await run_in_threadpool(utils.generate_content, db, payload.topic)
    return {"generated_text": generate_text}

@app.post("/analyze/")
async def analyze_content(payload: schemas.AnalyzePayload, db: Session = Depends(get_db)):
    readability, sentiment = await run_in_threadpool(utils.analyze_content, db, payload.content)
    return {"readability": readability, "sentiment": sentiment}


