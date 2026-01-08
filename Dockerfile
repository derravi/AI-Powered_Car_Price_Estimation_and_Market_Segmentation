FROM python:3.13.7

RUN mkdir -p /ai_powered_car_price_estimation_and_market_segmentation

WORKDIR /ai_powered_car_price_estimation_and_market_segmentation

COPY . /ai_powered_car_price_estimation_and_market_segmentation

RUN pip install -r requirements.txt

ENV PORT=8000

EXPOSE 8000

CMD [ "uvicorn","app:app","--host","0.0.0.0","--port","8000" ]