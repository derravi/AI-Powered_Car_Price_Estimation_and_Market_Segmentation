FROM python:3.13.7

RUN mkdir -p /AI_Powered_Car_Price_Estimation_and_Market_Segmentation

WORKDIR /AI_Powered_Car_Price_Estimation_and_Market_Segmentation

COPY . /AI_Powered_Car_Price_Estimation_and_Market_Segmentation

RUN pip install -r requirement.txt

ENV PORT=8000

EXPOSE 8000

CMD [ "uvicorn","app:app","--host","0.0.0.0","--port","8000" ]