from functools import lru_cache
from io import BytesIO
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from src.prediction.breast_cancer_predictor import BreastCancerPredictor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "breast_cancer_logistic_regression.joblib"
WEB_ROOT = Path(__file__).resolve().parent / "web"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_ROWS = 2000
IGNORED_COLUMNS = {"id", "diagnosis"}

app = FastAPI(
    title="Classificador experimental de tumores de mama",
    description=(
        "Demonstração acadêmica de classificação com o dataset "
        "Breast Cancer Wisconsin Diagnostic. Não é uma ferramenta clínica."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_predictor():
    return BreastCancerPredictor(MODEL_PATH)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_ROOT / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.is_file()}


@app.get("/api/template", include_in_schema=False)
def download_template():
    dataset_path = PROJECT_ROOT / "data" / "breast_cancer" / "breast_cancer.csv"
    if not dataset_path.is_file():
        raise HTTPException(
            status_code=503,
            detail="Dataset de exemplo não encontrado para gerar a planilha.",
        )

    dataframe = pd.read_csv(dataset_path)
    if "diagnosis" in dataframe.columns:
        dataframe = pd.concat([
            dataframe[dataframe["diagnosis"] == diagnosis].head(2)
            for diagnosis in ("B", "M")
        ])
    sample = dataframe.drop(
        columns=[column for column in ("id", "diagnosis", "Unnamed: 32")
                 if column in dataframe.columns]
    ).head(4)
    sample = sample.apply(
        lambda column: column.map(lambda value: format(value, ".15g"))
    )
    output = BytesIO(sample.to_csv(index=False, sep=",", decimal=".").encode("utf-8-sig"))

    return StreamingResponse(
        output,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="modelo_medidas.csv"'},
    )


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Envie um arquivo CSV (.csv) separado por vírgulas.",
        )

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="A planilha excede o limite de 5 MB.",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="A planilha está vazia.")

    try:
        header = contents.decode("utf-8-sig").splitlines()[0]
        if ";" in header and "," not in header:
            raise HTTPException(
                status_code=400,
                detail=(
                    "O CSV deve usar vírgula (,) para separar colunas "
                    "e ponto (.) como separador decimal."
                ),
            )
        dataframe = pd.read_csv(
            BytesIO(contents),
            sep=",",
            decimal=".",
            encoding="utf-8-sig",
        )
    except (ImportError, OSError, ValueError, IndexError) as error:
        raise HTTPException(
            status_code=400,
            detail="Não foi possível ler o arquivo. Verifique se é um CSV válido separado por vírgulas.",
        ) from error

    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    normalized_columns = [column.casefold() for column in dataframe.columns]
    if len(normalized_columns) != len(set(normalized_columns)):
        raise HTTPException(
            status_code=400,
            detail="A planilha contém nomes de coluna duplicados.",
        )

    if len(dataframe) > MAX_ROWS:
        raise HTTPException(
            status_code=413,
            detail=f"A planilha pode conter no máximo {MAX_ROWS} registros.",
        )

    metadata_columns = [
        column for column in dataframe.columns
        if column.casefold() in IGNORED_COLUMNS
        or column.casefold().startswith("unnamed:")
    ]
    row_ids = (
        dataframe["id"].fillna("").astype(str).tolist()
        if "id" in dataframe.columns
        else [str(index) for index in range(1, len(dataframe) + 1)]
    )
    features = dataframe.drop(columns=metadata_columns)

    try:
        predictor = get_predictor()
        predictions = predictor.predict(features)
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo treinado não encontrado. Execute `python -m src.main` "
                "na raiz do projeto antes de iniciar a aplicação."
            ),
        ) from error
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    results = []
    for index, prediction in predictions.iterrows():
        row_id = row_ids[index] if row_ids[index] else str(index + 1)
        results.append({
            "row": row_id,
            "diagnosis": prediction["diagnosis"],
            "probability_benign": float(prediction["probability_benign"]),
            "probability_malignant": float(prediction["probability_malignant"]),
        })

    benign = sum(result["diagnosis"] == "Benigno" for result in results)
    return {
        "total": len(results),
        "summary": {
            "benign": benign,
            "malignant": len(results) - benign,
        },
        "results": results,
    }
