from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

from ml_boston.constants import APP_HOST, APP_PORT
from ml_boston.pipline.prediction_pipeline import BostonData, BostonClassifierWithLocalModel
from ml_boston.pipline.training_pipeline import TrainPipeline

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.crim: Optional[str] = None
        self.zn: Optional[str] = None
        self.indus: Optional[str] = None
        self.chas: Optional[str] = None
        self.nox: Optional[str] = None
        self.rm: Optional[str] = None
        self.age: Optional[str] = None
        self.dis: Optional[str] = None
        self.rad: Optional[str] = None
        self.tax: Optional[str] = None
        self.ptratio: Optional[str] = None
        self.b: Optional[str] = None
        self.lstat: Optional[str] = None
   

    async def get_boston_data(self):
        form = await self.request.form()
        self.crim = float(form.get("crim"))
        self.zn = float(form.get("zn"))
        self.indus = float(form.get("indus"))
        self.chas = float(form.get("chas"))
        self.nox = float(form.get("nox"))
        self.rm = float(form.get("rm"))
        self.age = float(form.get("age"))
        self.dis = float(form.get("dis"))
        self.rad = float(form.get("rad"))
        self.tax = float(form.get("tax"))
        self.ptratio = float(form.get("ptratio"))
        self.b = float(form.get("b"))
        self.lstat = float(form.get("lstat"))


@app.get("/", tags=["authentication"])
async def index(request: Request):

    return templates.TemplateResponse(
            "boston.html",{"request": request, "context": "Rendering"})


@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_boston_data()
        boston_data = BostonData(
                                crim= float(form.crim),
                                zn = float(form.zn),
                                indus = float(form.indus),
                                chas = float(form.chas),
                                nox= float(form.nox),
                                rm= float(form.rm),
                                age = float(form.age),
                                dis= float(form.dis),
                                rad= float(form.rad),
                                tax= float(form.tax),
                                ptratio= float(form.ptratio),
                                b = float(form.b),
                                lstat = float(form.lstat)
                                )
        
        boston_df = boston_data.get_boston_input_data_frame()

        model_predictor = BostonClassifierWithLocalModel()

        value = model_predictor.predict(dataframe=boston_df)[0]

        return templates.TemplateResponse(
            "boston.html",
            {"request": request, "context": format(value, ".2f")},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}


if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)