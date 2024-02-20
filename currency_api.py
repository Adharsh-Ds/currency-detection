import os
import re
import cv2

from PIL import Image
import logging as logger
from fastapi import FastAPI
from pydantic.main import BaseModel
from core.main import  generate_tokens_mindee, process_ocr_output, extract_values, get_currency_ascii



app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Currency-Detection"}


class FileRequest(BaseModel):
    inputFilePath: str



@app.post("/api/currency-attribution")
async def get_currency_detection(request: FileRequest):

    logger.info("inbound currency attribution request {}".format(request.__str__()))

    list_of_ocr_result = []
    second_value = None
    confi_score = None
    try:
        ocr_output = generate_tokens_mindee(Image.open(request.inputFilePath))
        n, n_plus_1, n_less_1 = process_ocr_output(ocr_output)
        process = extract_values(n, n_plus_1, n_less_1)

        print(process)
        if len(process) > 0:
            items = list(process[0].items())
            if len(items) > 1:
                _, second_value = items[1]
                _,confi_score = items[2]
            else:
                second_value = None
                confi_score = None

        ascii_value_ = get_currency_ascii(process[1])

        output_json = {
            "filePath": request.inputFilePath,
            "currency_symbol": process[0]['symbol'],
            "value": second_value,
            "asciiValue": ascii_value_,
            "confidence_score": confi_score
        }

        list_of_ocr_result.append(output_json)

        return list_of_ocr_result

    except Exception as ex:
        raise ex


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.5", port=8000, reload=True)
