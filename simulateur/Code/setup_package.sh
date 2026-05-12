#!/bin/sh
FOLDER=${1%/} # %/ removes potential trailing "/"
if [ -z "$FOLDER" ]
then
	echo "need the name of the generated folder !"
	exit 1
fi
PACKAGE=PACKAGE_"${FOLDER/generated_/}"/ # remove "generated" from folder name
CONTAINER=container

echo "usage : bash setup_package.sh <generated_folder>"

rm -r $PACKAGE
mkdir $PACKAGE
cp $FOLDER/vitis_platform/dtbo_output/pl.dtbo $PACKAGE
cp $FOLDER/system_project/app_component/build/hw/app_component $PACKAGE
cp $FOLDER/system_project/system_project/build/hw/hw_link/*.xclbin $PACKAGE
echo '{\n  "shell_type" : "XRT_FLAT",\n  "num_slots": "1"\n}' > $PACKAGE/shell.json
chmod +x $PACKAGE/app_component


mkdir -p $PACKAGE/data && cp -r data/ $PACKAGE/

mkdir $PACKAGE/app
cp info.csv uvw_64_vec.csv $PACKAGE/app/
mv $PACKAGE/app_component  $PACKAGE/app
cp $PACKAGE/$CONTAINER.xclbin $PACKAGE/app/top_degridder.xclbin

mkdir $PACKAGE/output/

mkdir $PACKAGE/$PACKAGE
mv $PACKAGE/$CONTAINER.xclbin $PACKAGE/pl.dtbo $PACKAGE/shell.json $PACKAGE/$PACKAGE

cp $FOLDER/system_project/system_project/build/hw/hw_link/container/reports/container/imp/impl_1_full_util_placed.rpt $PACKAGE

echo "setup done !"