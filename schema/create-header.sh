#!/bin/bash -e

OUTPUT_H="mnist_schema_generated.h"
OUTPUT_F="./esp8266-tf-client/"

echo "Creating framebuffer C++ header..."
flatc -o ./schema/ --cpp ./schema/schema.fbs

echo "Copying ${OUTPUT_H} to ${OUTPUT_F}..."
mv ./schema/schema_generated.h ./${OUTPUT_F}/${OUTPUT_H}
