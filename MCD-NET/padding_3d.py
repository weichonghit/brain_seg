
import numpy as np
import time
import glob
import h5py
import numpy as np
import nibabel as nib
from os.path import join
from multiprocessing import Pool

def padding(subjects):
	
	for idx, current_subject in enumerate(subjects):
		start = time.time()
		# read data
		orig   = nib.load(join(current_subject, orig_name))
		affine = orig.affine.copy()  # save the affine and header
		hdr    = orig.header.copy() 
		orig   = np.asarray(orig.get_fdata())

		# read label
		aseg = nib.load(join(current_subject, aparc_name))
		aseg = np.asarray(aseg.get_fdata(), dtype ='uint16')
		aseg = map_aparc_aseg2label_level_4(aseg)

		new_orig = np.ndarray(shape=(256, 256, 256), dtype=np.float32)
		new_aseg = np.ndarray(shape=(256, 256, 256), dtype=np.uint16)
		w, h ,d = orig.shape
		
		print("Volume Nr: {}/{} Processing MRI Data from {}/{}".format(idx + 1, data_set_size, current_subject, orig_name))

		for i in range(w):
			for j in range(h):
				for k in range(d):
					new_orig[i,j,k] = orig[i,j,k]
					new_aseg[i,j,k] = aseg[i,j,k]

		# add affine and header to the padded data
		new_orig_nii = nib.Nifti1Image(new_orig, affine, hdr) 
		new_aseg_nii = nib.Nifti1Image(new_aseg, affine, hdr) 

		# save
		nib.save(new_orig_nii, join(current_subject,new_orig_name))
		nib.save(new_aseg_nii, join(current_subject,new_aparc_name))
		end = time.time() - start
		print("Volume: {} Finished Data padding in {:.3f} seconds.".format(idx+1, end))

if __name__=='__main__':
	# read data list
	data_path = '../data_val/'
	pattern = 'result*'
	search_pattern = join(data_path, pattern)
	subject_dirs = glob.glob(search_pattern)
	data_set_size = len(subject_dirs)
	print("\ntotal files: {}\n".format(data_set_size))
	

	# set origin data name
	orig_name  = 't1seg.nii'
	aparc_name = 't1seg_label.nii'

	# set output data name
	new_orig_name  = 'new_orig_nii.nii'
	new_aparc_name = 'new_aseg_nii.nii'
	
	# set core for multiprocessing
	core_num = 4
	start_d = time.time()
	child_subject_dirs = [[] for i in range(core_num)]
	
	for idx, current_subject in enumerate(subject_dirs):
		child_subject_dirs[idx%core_num].append(current_subject)
	print(np.shape(child_subject_dirs))
	p = Pool(core_num)
	for i in range(core_num):
		p.apply_async(padding(child_subject_dirs[i]), args=(i,))
	print('Waiting for all subprocesses done...')
	p.close()
	p.join()
	print('All subprocesses done.')

	end_d = time.time() - start_d
	print("Successfully padding {} volumes in {:.3f} seconds.".format(data_set_size, end_d))

