from typing import Dict, Union, List
import os
import sys
sys.path.append('DeepJuiceDev')
from deepjuice import *
from torchmetrics.functional import pearson_corrcoef
from deepjuice.benchmark import get_results_max
from deepjuice.alignment import get_scoring_method

def effective_dimensionality(feature_map):
    # first, we define a GPU-capable PCA
    pca = TorchPCA(device='cuda:0')
    
    # then fit the PCA...
    pca.fit(feature_map)
    
    # then extract the eigenspectrum
    eigvals = pca.explained_variance_
    
    # then return effective dimensionality (on CPU)
    return (eigvals.sum() ** 2 / (eigvals ** 2).sum()).item()

def get_benchmarking_results(benchmark, feature_extractor, 
                             layer_index_offset = 0,
                             metrics = ['crsa','ereg','ersa'],
                             rdm_distance = 'pearson',
                             rsa_distance = 'pearson',
                             score_types = ['pearsonr'],
                             stack_final_results = True,
                             feature_map_stats = None,
                             alpha_values = np.logspace(-1,5,7).tolist(), 
                             average_case = 0, target_depths = None,
                             device='auto'):
    
    # use a CUDA-capable device, if available, else: CPU
    if device == 'auto': device = get_device_id(device)
    # device='cpu'
    
    # record key information about each method for reference
    method_info = {'regression': {'encoding_model': 'ridge'}, 
                   'rsa': {'rdm_distance': rdm_distance,
                           'rsa_distance': rsa_distance}}
    
    # initialize an empty list to record scores over layers
    scoresheet_lists = {metric: [] for metric in metrics}

    # if no feature_map_stats provided, make empty dict:
    if feature_map_stats is None: feature_map_stats = {}
    
    # get the voxel (neuroid) indices for each specified roi
    roi_indices = benchmark.get_roi_indices(row_number=True)
    
    # put all the train-test RDMS on the GPU for fast compute
    target_rdms = apply_tensor_op(benchmark.splithalf_rdms, 
                                   lambda x: x.to(device))
    
    # convert the benchmark response_data to a tensor on GPU
    y = (convert_to_tensor(benchmark.response_data.to_numpy())
         .to(dtype=torch.float32, device=device))
    
    # split response data into train and test sets
    y = {'train': y[:,::2].T, 'test': y[:,1::2].T}
    
    # initialize the regression, in this case ridge regression with LOOCV over alphas
    regression = TorchRidgeGCV(alphas = alpha_values, device = device, scale_X = True)
    
    # initialize a dictionary of scoring metrics to apply to the predicted outputs
    score_funcs = {score_type: get_scoring_method(score_type) for score_type in score_types}
    
    # layer_index = 0 # keeps track of depth
    
    if average_case == 0:
        # access extractor_ids first
        extractor_uids = feature_extractor.get_uids()
        # Based on whether target_depths is None or not
        if target_depths:
            # It's a default setting, but please feel free to change
            target_depths_list = np.arange(0,1,0.1)
            # attain target_extractor_ids and target_layer_index
            target_layer_indexes =  np.floor(len(extractor_uids)*target_depths_list).astype(int)
            target_extractor_uids = [extractor_uids[i] for i in target_layer_indexes]
        else:
            target_layer_indexes = np.arange(len(extractor_uids))
            target_extractor_uids = extractor_uids
            
        # now, we iterate over our extractor
        for feature_maps in tqdm(feature_extractor, desc = 'Feature Extraction (Batch)'):
            # update feature_maps to remove those unrelated ones
            if target_depths:
                feature_maps = {key: feature_maps[key] for key in feature_maps if key in target_extractor_uids}
            
            # only these two metrics would require the calculations of rdms
            if 'crsa' in metrics or 'ersa' in metrics:
                # for each feature_map from the extractor batch, we first need to compute the traintest RDMS
                model_rdms = {uid: {'train': compute_rdm(feature_map[::2], method=rdm_distance, device=device),
                                    'test': compute_rdm(feature_map[1::2], method=rdm_distance, device=device)}
                            for uid, feature_map in tqdm(feature_maps.items(), desc = 'Making RDMs (Layer)')}
            
            # now, we loop over our batch of feature_maps from the extractor...
            # ...starting by defining an iterator that will track our progress
            feature_map_iterator = tqdm(feature_maps.items(), desc = 'Brain Mapping (Layer)')
            for i,(feature_map_uid, feature_map) in enumerate(feature_map_iterator):
                # 
                # layer_index += 1 # one layer deeper in feature_maps
                
                # main data to add to our scoresheet per feature_map
                # layer_index + layer_index_offset
                feature_map_info = {'model_layer': feature_map_uid, 
                                    # layer_index_offset is used here in case of subsetting
                                    'model_layer_index': target_layer_indexes[i]}
                
                # now, our X Variable: the splithalf feature_map on GPU
                feature_map = get_feature_map_srps(feature_map, device=device, progress=False)
                X = feature_map.squeeze().to(torch.float32).to(device)
                X = {'train': X[0::2,], 'test': X[1::2,:]} # splithalf
                
                # here, we calculate auxiliary stats on our feature_maps
                aux_stats = {split: {stat: stat_func(Xi) for stat, stat_func 
                                    in feature_map_stats.items()} for split, Xi in X.items()}

                regression.fit(X['train'], y['train']) # fit the regression on the train split
                
                # RidgeGCV gives us both internally generated LOOCV values for the train dataset
                # as well as the ability to predict our test set in the same way as any regressor
                y_pred = {'train': regression.cv_y_pred_, 'test': regression.predict(X['test'])}
                # loop over cRSA, SRPR, eRSA...
                for metric in scoresheet_lists:
                    # classical RSA score
                    if metric == 'crsa':       
                        for split in ['train', 'test']:
                            # get the relevant train-test split of the model RDM
                            model_rdm = model_rdms[feature_map_uid][split]
                            for region in benchmark.rdms:
                                for subj_id in benchmark.rdms[region]:
                                    # get the relevant train-test split of the brain RDM
                                    target_rdm = target_rdms[region][subj_id][split]
                                    
                                    # compare lower triangles of model + brain RDM
                                    # with our specified 2nd-order distance metric
                                    score = compare_rdms(model_rdm, target_rdm,
                                                        method = rsa_distance)
                                    
                                    # add the scores to a "scoresheet"
                                    scoresheet = {**feature_map_info,
                                                'region': region, 
                                                'subj_id': subj_id,
                                                'cv_split': split,
                                                'score': score, 
                                                **aux_stats[split],
                                                **method_info['rsa']}
                                    
                                    # append the scoresheet to our running list
                                    scoresheet_lists['crsa'].append(scoresheet)

                    # encoding model score
                    if metric == 'ereg':
                        for split in ['train','test']:
                            for score_type, score_func in score_funcs.items():
                                # calculate score per neuroid_id with score_type
                                scores = score_func(y[split], y_pred[split])
                                for region in benchmark.rdms:
                                    for subj_id in benchmark.rdms[region]:
                                        # get the response_indices for current ROI group
                                        response_indices = roi_indices[region][subj_id]
                                        
                                        # average the scores across the response_indices
                                        score = scores[response_indices].mean().item()
                                                                        
                                        # add the scores to a "scoresheet"
                                        scoresheet = {**feature_map_info,
                                                    'region': region, 
                                                    'subj_id': subj_id,
                                                    'cv_split': split,
                                                    'score': score, 
                                                    **aux_stats[split],
                                                    **method_info['regression']}
                                        
                                        # append the scoresheet to our running list
                                        scoresheet_lists['ereg'].append(scoresheet)
                                    
                    # encoding RSA score
                    if metric == 'ersa':
                        for split in ['train', 'test']:
                            for region in benchmark.rdms:
                                for subj_id in benchmark.rdms[region]:
                                    # get the relevant train-test split of the brain RDM
                                    target_rdm = target_rdms[region][subj_id][split]
                                    
                                    # get the response_indices for current ROI group
                                    response_indices = roi_indices[region][subj_id]
                                    
                                    # get predicted values for each response_index...
                                    y_pred_i = y_pred[split][:, response_indices]
                                    
                                    # ... and use them to calculate the weighted RDM
                                    model_rdm = compute_rdm(y_pred_i, rdm_distance)
                                    
                                    # compare brain-reweighted model RDM to brain RDM
                                    # with our specified 2nd-order distance metric...
                                    score = compare_rdms(model_rdm, target_rdm,
                                                        method = rsa_distance)
                                    
                                    # add the scores to a "scoresheet"
                                    scoresheet = {**feature_map_info,
                                                'region': region, 
                                                'subj_id': subj_id,
                                                'cv_split': split,
                                                'score': score, 
                                                **aux_stats[split],
                                                **method_info['rsa']}

                                    # append the scoresheet to our running list
                                    scoresheet_lists['ersa'].append(scoresheet) 
    else:
        #
        extractor_uids=feature_extractor[0].get_uids()
        # Based on whether target_depths is None or not
        if target_depths:
            # It's a default setting, but please feel free to change
            target_depths_list = np.arange(0,1,0.1)
            # attain target_extractor_ids and target_layer_index
            target_layer_indexes =  np.floor(len(extractor_uids)*target_depths_list).astype(int)
            target_extractor_uids = [extractor_uids[i] for i in target_layer_indexes]
        else:
            target_layer_indexes = np.arange(len(extractor_uids))
            target_extractor_uids = extractor_uids

        for i,extractor_uid in enumerate(target_extractor_uids):
            # layer_index += 1 # one layer deeper in feature_maps
            # main data to add to our scoresheet per feature_map
            feature_map_info = {'model_layer': extractor_uid, 
                                # layer_index_offset is used here in case of subsetting
                                'model_layer_index': target_layer_indexes[i]}      
            # now, our X Variable: the splithalf feature_map on GPU
            # print(get_feature_map_srps(feature_extractor_list[0].get(extractor_uid),device=device))
            feature_map_list = torch.stack([get_feature_map_srps(feature_extractor[i].get(extractor_uid), device=device) for i in range(len(feature_extractor))])
            # which image we want to choose
            maximal_number_of_layers_total = len(feature_map_list)
            maximal_number_of_layers_avg = maximal_number_of_layers_total-2
            number_of_experiments =  maximal_number_of_layers_total
            for number_of_layers in range(1,maximal_number_of_layers_avg):
                for count in range(number_of_experiments):
                    random_index = np.random.choice(range(maximal_number_of_layers_total), number_of_layers, replace =False)
                    # print(feature_map_list.shape)
                    # selected_feature_maps = [feature_map_list[i] for i in random_index]
                    # feature_map = torch.cat(selected_feature_maps, dim=1)
                    feature_map = torch.mean(feature_map_list[random_index,],0)
                    # add the model_rdms
                    model_rdms = {'train': compute_rdm(feature_map[::2], method=rdm_distance, device=device),
                                    'test': compute_rdm(feature_map[1::2], method=rdm_distance, device=device)}
                    # feature_map = torch.mean(torch.stack([feature_extractor_list[i].get(extractor_uid) for i in range(len(feature_extractor_list))]),0)
                    # print(feature_map)
                    # feature_map = get_feature_map_srps(feature_map,device=device)
                    X = feature_map.to(torch.float32).to(device)
                    X = {'train': X[0::2,], 'test': X[1::2,:]} # splithalf
                    
                    # here, we calculate auxiliary stats on our feature_maps
                    aux_stats = {split: {stat: stat_func(Xi) for stat, stat_func 
                                        in feature_map_stats.items()} for split, Xi in X.items()}

                    regression.fit(X['train'], y['train']) # fit the regression on the train split
                    
                    # RidgeGCV gives us both internally generated LOOCV values for the train dataset
                    # as well as the ability to predict our test set in the same way as any regressor
                    y_pred = {'train': regression.cv_y_pred_, 'test': regression.predict(X['test'])}
                    for metric in scoresheet_lists:
                        # classical RSA score
                        if metric == 'crsa':       
                            for split in ['train', 'test']:
                                # get the relevant train-test split of the model RDM
                                model_rdm = model_rdms[split]
                                for region in benchmark.rdms:
                                    for subj_id in benchmark.rdms[region]:
                                        # get the relevant train-test split of the brain RDM
                                        target_rdm = target_rdms[region][subj_id][split]
                                        
                                        # compare lower triangles of model + brain RDM
                                        # with our specified 2nd-order distance metric
                                        score = compare_rdms(model_rdm, target_rdm,
                                                            method = rsa_distance)
                                        
                                        # add the scores to a "scoresheet"
                                        scoresheet = {**feature_map_info,
                                                    'region': region, 
                                                    'subj_id': subj_id,
                                                    'cv_split': split,
                                                    'layer_number': number_of_layers,
                                                    'score': score, 
                                                    **aux_stats[split],
                                                    **method_info['rsa']}
                                        
                                        # append the scoresheet to our running list
                                        scoresheet_lists['crsa'].append(scoresheet)
                                        
                        # encoding model score
                        if metric == 'ereg':
                            for split in ['train','test']:
                                for score_type, score_func in score_funcs.items():
                                    # calculate score per neuroid_id with score_type
                                    scores = score_func(y[split], y_pred[split])
                                    for region in benchmark.rdms:
                                        for subj_id in benchmark.rdms[region]:
                                            # get the response_indices for current ROI group
                                            response_indices = roi_indices[region][subj_id]
                                            
                                            # average the scores across the response_indices
                                            score = scores[response_indices].mean().item()
                                                                            
                                            # add the scores to a "scoresheet"
                                            scoresheet = {**feature_map_info,
                                                        'region': region, 
                                                        'subj_id': subj_id,
                                                        'cv_split': split,
                                                        'layer_number': number_of_layers,
                                                        'score': score, 
                                                        **aux_stats[split],
                                                        **method_info['regression']}
                                            
                                            # append the scoresheet to our running list
                                            scoresheet_lists['ereg'].append(scoresheet)
                        
                        # encoding RSA score
                        if metric == 'ersa':
                            for split in ['train', 'test']:
                                for region in benchmark.rdms:
                                    for subj_id in benchmark.rdms[region]:
                                        # get the relevant train-test split of the brain RDM
                                        target_rdm = target_rdms[region][subj_id][split]
                                        
                                        # get the response_indices for current ROI group
                                        response_indices = roi_indices[region][subj_id]
                                        
                                        # get predicted values for each response_index...
                                        y_pred_i = y_pred[split][:, response_indices]
                                        
                                        # ... and use them to calculate the weighted RDM
                                        model_rdm = compute_rdm(y_pred_i, rdm_distance)
                                        
                                        # compare brain-reweighted model RDM to brain RDM
                                        # with our specified 2nd-order distance metric...
                                        score = compare_rdms(model_rdm, target_rdm,
                                                            method = rsa_distance)
                                        
                                        # add the scores to a "scoresheet"
                                        scoresheet = {**feature_map_info,
                                                    'region': region, 
                                                    'subj_id': subj_id,
                                                    'cv_split': split,
                                                    'layer_number': number_of_layers,
                                                    'score': score, 
                                                    **aux_stats[split],
                                                    **method_info['rsa']}

                                        # append the scoresheet to our running list
                                        scoresheet_lists['ersa'].append(scoresheet)                              
                           
    # return all the train-test RDMS tn the CPU after use
    apply_tensor_op(benchmark.splithalf_rdms, lambda x: x.to('cpu'))
                             
    # if we don't stack, results are a dictionary of dataframes...
    results = {metric: pd.DataFrame(scores) for metric, scores 
                in scoresheet_lists.items()}
    if stack_final_results:
        # if we do stack, results are a single concatenated dataframe
        # with only the common_columns of each (excluding method data)
        result_columns = pd.unique([col for results in results.values() 
                                    for col in results.columns]).tolist()

        common_columns = [col for col in result_columns if
                          all(col in result.columns for 
                              result in results.values())]

        common_columns = ['metric'] + common_columns # indicator

        results_list = []
        for metric, result in results.items():
            result.insert(0, 'metric', metric)
            results_list.append(result[common_columns])
            # ...if we do stack, results are a single dataframe

        new_results = pd.concat(results_list)   
    return pd.concat(results_list) if stack_final_results else results



# To organize the path for containing images
def process_image_type(base_path: str, config: Dict) -> Union[List[str], Dict[int, List[str]]]:
    image_dir = base_path
    benchmark = config['benchmark']
    image_extensions = '*.jpg'  # Assuming only JPEG images; adjust if necessary

    # Check if image_dir contains subdirectories
    subdirs = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    if subdirs:
        # Create a dictionary of image paths indexed by image ID
        img2img_paths = {
            img_id: sorted(glob(os.path.join(image_dir, str(img_id), image_extensions)))
            for img_id in benchmark.response_data.columns.astype(int).tolist()
        }
    else:
        # If no subdirectories, collect all image paths directly in image_dir
        img2img_paths = sorted(
            glob(os.path.join(image_dir, image_extensions)),
            key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
        )

    # Determine the return type based on whether subdirectories were found
    if subdirs:
        return img2img_paths  # Return a dictionary
    else:
        return img2img_paths  # Return a list

# The function is to make the table   
def result_table(image_kind: str, img2img_paths: Union[List[str], Dict[int, List[str]]], config: Dict) -> pd.DataFrame:
    results_list = []  # initialize and fill with original / variations
    device, benchmark, model, preprocess, metrics = config['device'], config['benchmark'], config['model'], config['preprocess'], config['metrics']
    average_case = config['average_case']
    stats = config['stats']
    maximal_num_of_images = config['maximal_num_of_images']
    target_depths = config['target_depths']
    
    if isinstance(img2img_paths, dict):
        num_sets = len(next(iter(img2img_paths.values())))  # get the length of the first list in the dict
        if maximal_num_of_images:
            if num_sets > maximal_num_of_images:
                num_sets = maximal_num_of_images
        if average_case == 0:
            # print(stats)
            for img_set in range(1, num_sets + 1):
                image_paths = [img_subset[img_set - 1] for img_subset in img2img_paths.values()]
                dataloader = get_data_loader(image_paths, preprocess)  # new image paths

                feature_extractor = FeatureExtractor(model, dataloader, flatten=True, initial_report=False, device=device)  # quiet start
                feature_extractor.modify_settings(batch_progress=True)  # dataloader batches
                results_list.append(get_benchmarking_results(benchmark, feature_extractor, 
                                                             layer_index_offset=0, metrics=metrics,
                                                             average_case=average_case,
                                                             feature_map_stats=stats,
                                                             target_depths=target_depths))

            results_set = deepcopy(results_list)
            for index, img_set in enumerate(range(1, num_sets + 1)):
                img_set = f'variation_{img_set}'
                results_set[index].insert(0, 'image_set', img_set)
        else:
            feature_extractor_list = []
            num_sets = len(next(iter(img2img_paths.values())))  # get the length of the first list in the dict
            # We can remote these functions if we can make it
            if maximal_num_of_images:
                if num_sets > maximal_num_of_images:
                    num_sets =maximal_num_of_images
            for img_set in range(1, num_sets + 1):
                image_paths = [img_subset[img_set - 1] for img_subset in img2img_paths.values()]
                
                dataloader = get_data_loader(image_paths, preprocess)  # our new image paths

                feature_extractor = FeatureExtractor(model, dataloader, flatten=True, initial_report=False, device=device)  # quiet start
                
                feature_extractor.modify_settings(batch_progress=True)  # dataloader batches

                feature_extractor_list.append(feature_extractor)
            # calculate
            results_list.append(get_benchmarking_results(benchmark, feature_extractor_list, 
                                                         layer_index_offset=0, metrics=metrics,
                                                         average_case=average_case,
                                                         feature_map_stats=stats,
                                                         target_depths=target_depths))
            results_set = deepcopy(results_list)
            for index, img_set in enumerate([average_case]):
                if isinstance(img_set, int):
                    img_set = f'average_case={img_set}'
                results_set[index].insert(0, 'image_set', img_set)
                
        results = pd.concat(results_set)
    else:
        if average_case != 0:
            raise ValueError('You seem to average calculate encoding score based on single copy of images. Not make sense')
        image_paths = img2img_paths
        dataloader = get_data_loader(image_paths, preprocess)  # new image paths

        feature_extractor = FeatureExtractor(model, dataloader, flatten=True, initial_report=False, device=device)  # quiet start
        feature_extractor.modify_settings(batch_progress=True)  
        
        feature_extractor.modify_settings(batch_progress=True)  # dataloader batches

        results_list.append(get_benchmarking_results(benchmark, feature_extractor, 
                                                     layer_index_offset=0, metrics=metrics,
                                                     feature_map_stats=stats,
                                                     target_depths=target_depths))
        results_set = deepcopy(results_list)

        img_set = 'original' if image_kind is None else 'variation_1'
        results_set[0].insert(0, 'image_set', img_set)

        results = pd.concat(results_set)

    return results

def get_feature_extractor_list(img2img_paths: Union[List[str], Dict[int, List[str]]], config: Dict) -> List:
    device, benchmark, model, preprocess = config['device'], config['benchmark'], config['model'], config['preprocess']
    maximal_num_of_images = config['maximal_num_of_images']
    
    feature_extractor_list = []
    
    if isinstance(img2img_paths, dict):
        num_sets = len(next(iter(img2img_paths.values())))  # get the length of the first list in the dict
        # print(num_sets)
        # Limit the number of sets if maximal_num_of_images is specified
        if maximal_num_of_images and num_sets > maximal_num_of_images:
            num_sets = maximal_num_of_images
        for img_set in range(1, num_sets + 1):
            image_paths = [img_subset[img_set - 1] for img_subset in img2img_paths.values()]
            dataloader = get_data_loader(image_paths, preprocess)
            # Create and configure feature extractor for each image set
            feature_extractor = FeatureExtractor(model, dataloader, flatten=True, initial_report=False, device=device)
            feature_extractor.modify_settings(batch_progress=True)
            
            feature_extractor_list.append(feature_extractor)
    else:
        image_paths = img2img_paths
        
        dataloader = get_data_loader(image_paths, preprocess)
        
        # Create and configure a single feature extractor
        feature_extractor = FeatureExtractor(model, dataloader, flatten=True, initial_report=False, device=device)
        feature_extractor.modify_settings(batch_progress=True)
        
        feature_extractor_list.append(feature_extractor)
    
    return feature_extractor_list


def load_feature_maps(benchmark, 
                      feature_extractor_list, 
                      config,
                      layer_index_offset = 0,
                      device='auto'):
    # use a CUDA-capable device, if available, else: CPU
    if device == 'auto': device = get_device_id(device)
    # Load the default parameter
    save = config['save']
    save_srpr = config['save_srpr']
    model_uid = config['model_uid']
    # now, we iterate over our extractor
    feature_map_dict = {}
    extractor_uids=feature_extractor_list[0].get_uids()
    # loop extractor uids
    for index, feature_extractor in enumerate(feature_extractor_list):
        layer_index = 0 # keeps track of depth
        for extractor_uid in extractor_uids:
            layer_index += 1 # one layer deeper in feature_maps
            # The information for model
            feature_map_dict['model_uid'] = model_uid
            feature_map_dict['img_set'] = f'variation_{index+1}'
            feature_map_dict['model_layer'] = extractor_uid
            feature_map_dict['model_layer_index'] = layer_index + layer_index_offset
            # now, our X Variable: the splithalf feature_map on GPU
            if  save:
                feature_map_original = feature_extractor_list[index].get(extractor_uid).squeeze().to(torch.float32).cpu()
                feature_map_dict['feature_map'] = feature_map_original
            if  save_srpr:
                feature_map = feature_extractor_list[index].get(extractor_uid)
                feature_map = get_feature_map_srps(feature_map, device=device, progress=False)
                feature_map_dict['feature_map_srpr'] = feature_map.squeeze().to(torch.float32).cpu()
    return feature_map_dict


def trim_lists_in_dict(original_dict, desired_length=20):
    trimmed_dict = {}
    for key, value in original_dict.items():
        if len(value) > desired_length:
            trimmed_dict[key] = list(np.random.choice(value, desired_length, replace=False))
        else:
            trimmed_dict[key] = value
    return trimmed_dict

# Function to merge dictionaries
def merge_dictionaries(dict_list):
    merged_dict = {}
    for d in dict_list:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = []  # Initialize the list if the key is not in the dictionary
            merged_dict[key]= merged_dict[key]+value
    return merged_dict
