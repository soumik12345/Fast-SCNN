from glob import glob


def check_validity(image_list, mask_list):
    '''Check Validity of Cityscapes Dataset
    Params:
        image_list -> List of Image files
        maks_list  -> List of Mask files
    '''
    for i in range(len(image_list)):
        assert image_list[i].split('/')[-1].split('_leftImg8bit')[0] == mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]
    for i in range(len(val_image_list)):
        assert val_image_list[i].split('/')[-1].split('_leftImg8bit')[0] == val_mask_list[i].split('/')[-1].split('_gtFine_labelIds')[0]
    print('All Right!')