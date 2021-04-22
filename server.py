from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from summary_model import Summary

class Item(BaseModel):
	rawText:str
	brif:float

app = FastAPI()

origins = [
	"http://localhost.tiangolo.com",
	"https://localhost.tiangolo.com",
	"http://localhost",
	"http://localhost:3000",
]

app.add_middleware(
	CORSMiddleware,
	allow_origins=origins,
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

@app.post("/resummer")
async def resummer(item:Item):
	summary = Summary(text=item.rawText, brif=item.brif)
	return summary