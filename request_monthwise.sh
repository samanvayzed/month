#!/bin/bash

curl -d ' [{"m":5 ,"y":2019}]' -H "Content-Type: application/json" -X POST http://localhost:12345/predict 


