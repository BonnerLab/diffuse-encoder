# import deepjuice
import os 
import sys
sys.path.append(os.getcwd())
from src.lib.deepjuice_env.utility import process_image_type, result_table,effective_dimensionality,trim_lists_in_dict,merge_dictionaries
import argparse
sys.path.append('DeepJuiceDev')
from deepjuice import *
from juicyfruits import NSDBenchmark,NSDBenchmarkCustom
import torchvision.models as models


def construct_path(*components):
    """
    Construct a path by joining non-None components.
    
    :param components: List of path components
    :return: A string representing the constructed path
    """
    non_none_components = [component for component in components if component is not None]
    if non_none_components:
        return os.path.join(*non_none_components)
    else:
        return None

# This function help to save results to csv
def save_results(results, result_dir, file_name):
    os.makedirs(result_dir, exist_ok=True)
    results.to_csv(os.path.join(result_dir, file_name), index=False)
    
    
def encoding_and_save(config: dict):
    image_path = config['image_path']
    result_path = config['result_path']
    option = config['option']

    # Option 'global'
    if option == 'global':
        if not os.path.exists(image_path) or not os.path.isdir(image_path):
            print(f"Directory '{image_path}' does not exist or is not a directory.")
            return
        image_types = [d for d in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, d))]
        for image_type in image_types:
            img2img_paths = process_image_type(os.path.join(image_path,image_type), config)
            results = result_table(image_type, img2img_paths, config)
            # save the results as csv file
            result_dir = os.path.join(result_path, config['model_uid']) if config['model_uid'] else result_path
            save_results(results, result_path, f'{image_type}.csv')

    elif option =='mixture':
        # Check if the provided path exists and is a directory
        if not os.path.exists(image_path) or not os.path.isdir(image_path):
            print(f"Directory '{image_path}' does not exist or is not a directory.")
            return

        # Process the initial image path
        img2img_paths = process_image_type(image_path, config)

        # Ask how many mixtures the user expects and validate the input
        num_of_mixtures = input('Since you indicate that you want to mix images, please specify the number of cases you want to mixture:\n')
        # num_of_mixtures = 3
        try:
            num_of_mixtures = int(num_of_mixtures)
            if num_of_mixtures < 1:
                raise ValueError
        except ValueError:
            print('Your input number is not valid or smaller than 1')
            return

        # Initialize the list with trimmed paths
        img2img_paths_list = [trim_lists_in_dict(img2img_paths)]

        # Collect paths for the specified number of mixtures
        '''
        image_path_list = ['/data/yrong12/experiment_3/kandinsky-2-2-decode/Positive_Text/s=1_g=7.5',
                           '/data/yrong12/experiment_3/sd-1-5/Positive_Text/s=1_g=7.5',
                           '/data/yrong12/experiment_3/sd-2/Positive_Text/s=1_g=7.5']
       image_path = image_path_list[i]
       '''
        for i in range(num_of_mixtures):
            print(f'This is the {i+1} image path you add:\n')
            image_path = input('Please specify the prompt you want to add: ')
            if not os.path.exists(image_path) or not os.path.isdir(image_path):
                print(f"Directory '{image_path}' does not exist or is not a directory.")
                continue
            
            img2img_paths = process_image_type(image_path, config)
            img2img_paths_list.append(trim_lists_in_dict(img2img_paths))

        # Merge the dictionaries containing image paths
        img2img_paths = merge_dictionaries(img2img_paths_list)

        # Set the image type and generate the result table
        image_type = 'mixture'
        results = result_table(image_type, img2img_paths, config)

        # Determine the result directory
        result_dir = os.path.join(result_path, config.get('model_uid', '')) if config.get('model_uid') else result_path

        # Save the results
        save_results(results, result_dir, f'{image_type}.csv')
            
    # Option 'base_score'
    elif option == 'base_score':
        img2img_paths = config['benchmark'].image_paths 
        results = result_table(None, img2img_paths, config)
        # save the results as csv file
        # result_dir = os.path.join(result_path,'benchmark')
        result_dir = result_path
        save_results(results, result_dir, f'benchmark_{config["model_uid"]}.csv')
    # Option for saving individual image types
    else:
        if not os.path.exists(image_path) or not os.path.isdir(image_path):
            print(f"Directory '{image_path}' does not exist or is not a directory.")
            return
        img2img_paths = process_image_type(image_path, config)
        # by default, if it's the local case
        # we simple consider that image_type would be the last folder name
        image_type = image_path.split('/')[-1]
        results = result_table(image_type, img2img_paths, config)
        result_dir = os.path.join(result_path, config['model_uid']) if config['model_uid'] else result_path
        save_results(results, result_path, f'{image_type}.csv')
        

def main():
    parser = argparse.ArgumentParser(description='Process image storage and result paths and do encoding for brain in DeepJuice.')
    parser.add_argument('--image_path', default= None, help='Path to the image storage directory.')
    parser.add_argument('--experiment_case', default=None, help='The experiment case identifier.')
    parser.add_argument('--generative_model', default=None, help='The type of generative model for utilize in this experiment')
    parser.add_argument('--image_type', default=None, help='Type of images to process.')
    parser.add_argument('--result_path', type=str, default='result', help='Path to the result storage directory.')
    parser.add_argument('--model_uid', default=None, help='Model UID.')
    parser.add_argument('--option', default='base_score', choices=['global', 'base_score', 'mixture','specific'], help='Option to specify the processing mode.')
    parser.add_argument('--device', default='cuda:0', help='Device to use for computation.')
    parser.add_argument('--metrics', nargs='+', default=['ereg'], help='You can test several metrics together')
    parser.add_argument('--average_case',type=int,default=0,help='0: No average. 1:average')
    parser.add_argument('--subject_id',type = str,help='The id folder you can specify for subjects')
    parser.add_argument('--trial_id',type = str,default='trial=0',help='The id folder you can specify for trial')
    parser.add_argument('--maximal_num_of_images',type = int,help='Maximal_num_of_images for feature extraction')
    parser.add_argument('--stats',default=0,help='You will add more results for stats. For example, effective dimensionality')
    parser.add_argument('--benchmark_path',help='You can supply the path to beta and meta for your benchmark')
    parser.add_argument('--target_depths',default = 0, help = 'If 0, we analyze all layers. If 1, the default would be 0:1:0.1')
    parser.add_argument('--model_state_dict_path',help = 'If you have some state dict you would like to import from outside, name its path')
    
    args = parser.parse_args()
    
    # Setup the benchmark
    # modify this path accordingly, pointing
    # Default: write the benchmark to my own beta and meta so that I can run it smoothly
    # when it need to publish, I should change this function here
    # write on some information for run by myself
    if not args.subject_id:
        args.subject_id = None
    if not args.trial_id:
        args.trial_id = 'trial=0'
        
    if not args.benchmark_path:
        dataset_path = '/home/yrong12/project/diffuse-encoder/DeepJuiceDev/juicyfruits/nsd_subset'

        # also, modify the path to the response data with your new path
        file_paths = {'response_data': f'{dataset_path}/response/{args.trial_id}/voxel_betas.csv',
                    'response_meta': f'{dataset_path}/response/{args.trial_id}/voxel_metas.csv',
                    'stimulus_data': f'{dataset_path}/stimulus/shared1000.csv'}

        # specify all metas that are NOT ROIs
        metadata_keys = ['subj_id', 'ncsnr']

        benchmark = NSDBenchmarkCustom(file_paths, metadata_keys)

        # point benchmark to the proper stimulus paths:
        image_root = f'{dataset_path}/stimulus/shared1000'
        benchmark.add_image_data(image_root = image_root)
        benchmark.build_rdms(compute_pearson_rdm)
    else:
        benchamrk=NSDBenchmark()
        benchmark.build_rdms(compute_pearson_rdm)


    # Load the model
    # model_uid = args.model_uid if args.model_uid else 'torchvision_alexnet_imagenet1k_v1'
    model_uid = args.model_uid if args.model_uid else 'torchvision_resnet50_imagenet1k_v1'
    if args.model_state_dict_path:
        # For other experiment cases, our analysis is based on Resnet50
        # Therefore, we would utilize the preprocessed file for Resnet50
        # Also, please load the default state dict file
        model, preprocess = get_deepjuice_model('torchvision_resnet50_imagenet1k_v1')
        # Load the standard ResNet-50 model
        model = models.resnet50()
        # Load the model parameters
        model.load_state_dict(torch.load(args.model_state_dict_path))
    else:
        try:
            model, preprocess = get_deepjuice_model(model_uid)
        except:
            # random weight Alexnet
            # Load AlexNet with random weights
            print('The result will be based on a random weight Alexnet')
            model, preprocess = get_deepjuice_model('torchvision_alexnet_imagenet1k_v1')
            alexnet = models.alexnet(pretrained=False)
            model = alexnet
    
    #
    image_path = construct_path(args.image_path, args.experiment_case, args.generative_model, args.image_type)
    
    if args.option =='base_score':
        result_path = construct_path(args.result_path,'benchmark',args.subject_id, args.trial_id)
    elif args.option == 'mixture':
        result_path = construct_path(args.result_path, args.experiment_case, model_uid, args.subject_id, args.trial_id)
    else:
        result_path = construct_path(args.result_path, args.experiment_case, model_uid, args.subject_id, args.trial_id,args.generative_model, args.image_type)
    # default to be effective dimensionaliy, you can change the code for updating new methods
    if args.stats:
        stats = {'effective_dimensionality': effective_dimensionality}
    else:
        stats = None
    # Configuration dictionary
    config = {
        'image_path': image_path,
        'result_path': result_path,
        'device': args.device,
        'average_case':args.average_case,
        'benchmark': benchmark,
        'model': model,
        'preprocess': preprocess,
        'model_uid': model_uid,
        'metrics': args.metrics,
        'option':args.option,
        'maximal_num_of_images':args.maximal_num_of_images,
        'stats':stats,
        'target_depths':args.target_depths
    }

    # Save CSV
    encoding_and_save(config)

if __name__ == '__main__':
    main()