#needs to specify ID and date on top here
#as well as creating cropped sequential camera images in a folder called cam
#needs to specify ID and date at the bottom for registration

## Getting images
cd /Volumes/storinator/scans
# cd ~/scans
ID='ADRC_61'
date='2023.01.09-23.13.44'
CW_ID='CW22-80'

T1='MP2RAGE_UNI_Images'
T2='T2_SPC'
num=$(ls -d PMB_ADRC/$date/$ID/$T2* | cut -d. -f6 | sort -n | sed "4q;d")

dst='/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/03-PMB'
echo rsync -a -R PMB_ADRC/$date/$ID/$T1* $dst
rsync -a -R PMB_ADRC/$date/$ID/$T1* $dst
echo rsync -a -R PMB_ADRC/$date/$ID/$T2*.$num $dst
rsync -a -R PMB_ADRC/$date/$ID/$T2*.$num $dst

cd $dst/PMB_ADRC/$date/$ID
mkdir mri

cd $dst/PMB_ADRC/$date/$ID/$T1*
dcm2niix -z y *
mv *.nii.gz $dst/PMB_ADRC/$date/$ID/mri/T1.nii.gz

cd $dst/PMB_ADRC/$date/$ID/$T2*
dcm2niix -z y *
mv *.nii.gz $dst/PMB_ADRC/$date/$ID/mri/T2.nii.gz

### nii2png
path='/Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/03-PMB/PMB_ADRC'

cd /Users/jinghangli/Library/CloudStorage/OneDrive-UniversityofPittsburgh/03-PMB/alignment_code

files=$(ls $path/$date/$ID/mri/T1.nii.gz)
for file in $files
do
    tag='i'
    input=$file
    output=$path/$date/$ID/T1_png
    mkdir $output
    echo $pwd
    python3 nii2png2.py $input $output $tag
    wait;
done

## remove background
input=$path/$date/$ID/T1_png
output=$path/$date/$ID/rmbg_png

mkdir $output
rembg p $input $output

## segBlob here
python3 segBlob.py $path $date $ID $CW_ID

cam_input=$path/$date/$ID/cam
cam_output=$path/$date/$ID/rembg_cam

mkdir $cam_output
rembg p $cam_input $cam_output

## Nuyl adjust images
# source activate /Users/jinghangli/miniforge3/envs/pytorch
# source activate /opt/miniconda3/envs/pytorch
# T1_img_path=$cam_output
# python3 nuyl_adjust_T1_06.py $T1_img_path

## png2nii
python3 png2nii.py $path $date $ID

## Registration
python3 Registration.py $path $date $ID 'Rigid'