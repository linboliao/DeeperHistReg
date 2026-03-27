import argparse
import os
from pathlib import Path
import shutil
from datetime import datetime
from typing import Union

import torch

from deeperhistreg.dhr_pipeline.registration_params import *
from deeperhistreg.dhr_pipeline import full_resolution as fr
from deeperhistreg.dhr_pipeline import registration_params as rp


def reg_config(source_path, target_path, output_path, tmp_path, gpu):
    registration_params: dict = default_nonrigid_fast()
    registration_params.update({"device": f"cuda:{gpu}"})
    registration_params['nonrigid_registration_params']['device'] = f"cuda:{gpu}"

    save_displacement_field: bool = False
    copy_target: bool = False
    delete_temporary_results: bool = True
    case_name: str = "Nonrigid"
    temporary_path: Union[str, Path] = tmp_path

    config = dict()
    config['source_path'] = source_path
    config['target_path'] = target_path
    config['output_path'] = output_path
    config['registration_parameters'] = registration_params
    config['case_name'] = case_name
    config['save_displacement_field'] = save_displacement_field
    config['copy_target'] = copy_target
    config['delete_temporary_results'] = delete_temporary_results
    config['temporary_path'] = temporary_path
    return config


def run(**config):
    try:
        registration_parameters_path = config['registration_parameters_path']
        registration_parameters = rp.load_parameters(registration_parameters_path)
    except KeyError:
        registration_parameters = config['registration_parameters']

    source_path = config['source_path']
    target_path = config['target_path']
    output_path = config['output_path']
    experiment_name = config['case_name']
    save_path = config['temporary_path']
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), str(datetime.now()))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    try:
        registration_parameters['logging_path'] = Path(save_path) / "logs.txt"
        registration_parameters['case_name'] = experiment_name
        pipeline = fr.DeeperHistReg_FullResolution(registration_parameters)
        pipeline.run_registration(source_path, target_path, save_path)
    except Exception as e:
        print(f"Exception: {e}")

    ### Copy Outputs and Clean ###
    if registration_parameters['save_final_images']:
        try:
            warped_name = [item for item in os.listdir(Path(save_path) / experiment_name / "Results_Final") if "warped_source" in item][0]
            shutil.copy(Path(save_path) / experiment_name / "Results_Final" / warped_name, Path(output_path) / warped_name)
            shutil.copy(Path(save_path) / "logs.txt", Path(output_path) / "logs.txt")
            if config['copy_target']:
                _, target_name = os.path.split(target_path)
                _, extension = os.path.splitext(target_name)
                target_name = "target" + extension
                shutil.copy(target_path, Path(output_path) / target_name)
        except Exception as e:
            print(f"Exception: {e}")

    if config['save_displacement_field']:
        shutil.copy(Path(save_path) / experiment_name / "Results_Final" / "displacement_field.mha", Path(output_path) / "displacement_field.mha")
        shutil.copy(Path(save_path) / experiment_name / "Results_Final" / "postprocessing_params.json", Path(output_path) / "postprocessing_params.json")

    if config['delete_temporary_results']:
        try:
            shutil.rmtree(save_path)
        except Exception as e:
            print(f"Exception: {e}")


parser = argparse.ArgumentParser(description="DeeperHistReg arguments")

parser.add_argument('--source', type=str, default='/NAS145/liaolinbo/Data/免疫治疗省肿瘤/60例胃癌/CD31/', help="Path to the source image")
parser.add_argument('--target', type=str, default='/NAS145/liaolinbo/Data/免疫治疗省肿瘤/60例胃癌/HE/', help="Path to the target image")
parser.add_argument('--output', type=str, default='/NAS145/liaolinbo/Data/免疫治疗省肿瘤/60例胃癌/DHR/', help="Path to the output folder")
args = parser.parse_args()
if __name__ == "__main__":
    source_dir, target_dir, output_dir = args.source, args.target, args.output
    for slide in os.listdir(source_dir):
        if os.path.exists(os.path.join(args.source, slide)):
            src = os.path.join(source_dir, slide)
            target = os.path.join(target_dir, slide)
            output = output_dir
            tmp = os.path.join(output_dir, os.path.splitext(slide)[0])
            cfg = reg_config(src, target, output, tmp, gpu=0)
            run(**cfg)
            mode = 'CD31'
            os.makedirs(os.path.join(output_dir, f'{mode}'), exist_ok=True)
            torch.cuda.empty_cache()
            try:
                shutil.move(os.path.join(output_dir, 'warped_source.tiff'), os.path.join(output_dir, f'{mode}/{slide}'))
                shutil.rmtree(tmp)
            except Exception as e:
                print(f"Exception: {e}")
