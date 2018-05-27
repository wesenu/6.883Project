import numpy as np
clean=np.load('WB-B-t0-clean.npy')
adv=np.load('WB-B-t0-adv.npy')
label=np.load('reconstruct/rec_t0.npy')
train_clean=clean[:10]
train_adv=adv[:10]
test_lab=label[:10]

np.save('vis/clean_0.npy',train_clean)
np.save('vis/adv_0.npy',train_adv)
np.save('vis/res_0.npy',test_lab)

clean=np.load('WB-B-t5-clean.npy')
adv=np.load('WB-B-t5-adv.npy')
label=np.load('reconstruct/rec_t5.npy')
train_clean=clean[:10]
train_adv=adv[:10]
test_lab=label[:10]

np.save('vis/clean_5.npy',train_clean)
np.save('vis/adv_5.npy',train_adv)
np.save('vis/res_5.npy',test_lab)


