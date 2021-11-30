#!/bin/zsh

### Author: Logan Z. J. Williams, 12th October 2021

### Example script for processing data for geometric deep learning benchmarking ###
### In this example, we are working with cortical metrics in TEMPLATE space ###

### PREREQUISITES for this script are: 
#### 1) Connectome Workbench, available here: https://www.humanconnectome.org/software/get-connectome-workbench 
#### 2) dHCP symmetric template, available here: https://brain-development.org/brain-atlases/atlases-from-the-dhcp-project/cortical-surface-template/ 
#### 3) Sixth-order icosphere, available in the icosahedrons folder (ico-6.surf.gii)
#### 4) Sixth-order icosphere warps, available at: https://emckclac-my.sharepoint.com/:f:/g/personal/k1812201_kcl_ac_uk/EluWzKNeKd5CmMqGc1n1cKcBwe2n2yU7CJrzoD_0u8r_7g


### loop over all subjects in list ###

for subid sesid in dHCP_subjects ; do 

### here dHCP_subjects is a csv file of size Nx2, where N = number of subjects, and the delimiter is a space or tab ### 

### loop over both hemispheres ###

for hemi in left right ; do 

### to generate all warps for each subject, loop over 100 warps provided in geometric-deep-learning-benchmarking repository (indexed from 0 to 99)

for num in {0..99} ; do 

### list variables - user will need to specify location of top directories ###
##users must specify path, wb_command path, location of the ico6 file, location of the warped ico6 files, path to the dHCP symmetric template##


path=/<path to dHCP data>/derivatives/dhcp_anat_pipeline/sub-${subid}/ses-${sesid} ### User needs to specify ###
myelin=${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_myelinmap.shape.gii
curvature=${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_curv.shape.gii
corr_thickness=${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_desc-corr_thickness.shape.gii
sulc=${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_sulc.shape.gii
merge_output=${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_merged.shape.gii
wb_command=<path to wb_command binary> ### User needs to specify ###
native_sphere=${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_sphere.surf.gii
ico6=/<path to sixth order icosphere>/ico-6.surf.gii ### User needs to specify ###
warped_ico6=/<path to icosphere warps>/ico-6_warp_${num}.surf.gii ### User needs to specify ###
resample_output=${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_merged_resample.shape.gii
warp_output=${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_merged_resample_warp_${num}.shape.gii 
xfm_sphere=${path}/xfm/sub-${subid}_ses-${sesid}_hemi-${hemi}_from-native_to-dhcpSym40_dens-32k_mode-sphere.surf.gii
native_midthickness=${pa${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_merged.shape.giith}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_midthickness.surf.gii
template_midthickness=<path to dHCP symmetric template>/dhcpSym_template/week-40_hemi-${hemi}_space-dhcpSym_dens-32k_midthickness.surf.gii ### User needs to specify ###
template_sphere=<path to dHCP symmetric template>/dhcpSym_template/week-40_hemi-${hemi}_space-dhcpSym_dens-32k_sphere.surf.gii
merge_output_template=${path}/anat/sub-${subid}_ses-${sesid}_hemi-${hemi}_dhcpSym40_merged.shape.gii


### merge metric files ### 
wb_command -metric-merge ${merge_output} -metric ${myelin} -metric ${curvature} -metric ${corr_thickness} -metric ${sulc}


### warp subject data from native space to dhcpSym40 ###

wb_command -metric-resample ${merge_output} ${xfm_sphere} ${template_sphere} ADAP_BARY_AREA ${merge_output_template} -area-surfs ${native_midthickness} ${template_midthickness} 

### resample merged metric file to sixth order icosphere ###

wb_command -metric-resample ${merge_output_template} ${template_sphere} ${ico6} BARYCENTRIC ${resample_output}

### resample from regular sixth order icosphere to warped sixth order icosphere ###

wb_command -metric-resample ${resample_output} ${ico6} ${warped_ico6} BARYCENTRIC ${warped_output} 

done

done

done

