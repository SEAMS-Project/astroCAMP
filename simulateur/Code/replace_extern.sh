#!/bin/bash

for var in "$@"
do
	sed -i -E '/Shared/ s/^([[:space:]]*)float/\1extern float/' $var
	sed -i -E '/Shared/ s/^([[:space:]]*)unsigned/\1extern unsigned/' $var
	sed -i -E '/Shared/ s/^([[:space:]]*)int/\1extern int/' $var
	sed -i -E '/Shared/ s/^([[:space:]]*)std::complex<float>/\1extern std::complex<float>/' $var
	sed -i -E '/Shared/ s/^([[:space:]]*)Config/\1extern Config/' $var
	sed -i -E '/Shared/ s/^([[:space:]]*)char/\1extern char/' $var
done