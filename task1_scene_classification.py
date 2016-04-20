#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# DCASE 2016::Acoustic Scene Classification / Baseline System

import matplotlib
matplotlib.use('Agg')

from src.ui import *
from src.general import *
from src.files import *

from src.features import *
from src.dataset import *
from src.evaluation import *

from src.keras_rnn import RNN
from src.extend_dataset import extend_dataset

import numpy
import csv
import argparse
import textwrap

from joblib import Parallel, delayed

from sklearn import mixture
from sklearn import preprocessing

__version_info__ = ('1', '0', '0')
__version__ = '.'.join(__version_info__)


def main(argv):
    numpy.random.seed(123456)  # let's make randomization predictable

    parser = argparse.ArgumentParser(
        prefix_chars='-+',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
            DCASE 2016
            Task 1: Acoustic Scene Classification
            Baseline system
            ---------------------------------------------
                Tampere University of Technology / Audio Research Group
                Author:  Toni Heittola ( toni.heittola@tut.fi )

            System description
                This is an baseline implementation for D-CASE 2016 challenge acoustic scene classification task.
                Features: MFCC (static+delta+acceleration)
                Classifier: GMM

        '''))

    # Setup argument handling
    parser.add_argument("-development", help="Use the system in the development mode", action='store_true',
                        default=False, dest='development')
    parser.add_argument("-challenge", help="Use the system in the challenge mode", action='store_true',
                        default=False, dest='challenge')

    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + __version__)
    args = parser.parse_args()

    # Load parameters from config file
    parameter_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  os.path.splitext(os.path.basename(__file__))[0]+'.yaml')
    params = load_parameters(parameter_file)
    params = process_parameters(params)
    make_folders(params)

    title("DCASE 2016::Acoustic Scene Classification / Baseline System")

    # Check if mode is defined
    if not (args.development or args.challenge):
        args.development = True
        args.challenge = False

    dataset_evaluation_mode = 'folds'
    if args.development and not args.challenge:
        print "Running system in development mode"
        dataset_evaluation_mode = 'folds'
    elif not args.development and args.challenge:
        print "Running system in challenge mode"
        dataset_evaluation_mode = 'full'

    # Get dataset container class
    dataset = eval(params['general']['development_dataset'])(data_path=params['path']['data'])

    # Fetch data over internet and setup the data
    # ==================================================
    if params['flow']['initialize']:
        dataset.fetch()

    # Extract features for all audio files in the dataset
    # ==================================================
    if params['flow']['extract_features']:
        section_header('Feature extraction')

        # Collect files in train sets
        files = []
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            for item_id, item in enumerate(dataset.train(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
            for item_id, item in enumerate(dataset.test(fold)):
                if item['file'] not in files:
                    files.append(item['file'])
        files = sorted(files)

        # Go through files and make sure all features are extracted
        do_feature_extraction(files=files,
                              dataset=dataset,
                              feature_path=params['path']['features'],
                              params=params['features'],
                              overwrite=params['general']['overwrite'])

        foot()

    # Prepare feature normalizers
    # ==================================================
    if params['flow']['feature_normalizer']:
        section_header('Feature normalizer')

        do_feature_normalization(dataset=dataset,
                                 feature_normalizer_path=params['path']['feature_normalizers'],
                                 feature_path=params['path']['features'],
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 overwrite=params['general']['overwrite'])

        foot()

    # System training
    # ==================================================
    if params['flow']['train_system']:
        section_header('System training')

        do_system_training(dataset=dataset,
                           model_path=params['path']['models'],
                           feature_normalizer_path=params['path']['feature_normalizers'],
                           feature_path=params['path']['features'],
                           classifier_params=params['classifier']['parameters'],
                           classifier_method=params['classifier']['method'],
                           dataset_evaluation_mode=dataset_evaluation_mode,
                           overwrite=params['general']['overwrite'],
                           feature_params=params['features']
                           )

        foot()

    # System evaluation in development mode
    if args.development and not args.challenge:

        # System testing
        # ==================================================
        if params['flow']['test_system']:
            section_header('System testing')

            do_system_testing(dataset=dataset,
                              feature_path=params['path']['features'],
                              result_path=params['path']['results'],
                              model_path=params['path']['models'],
                              feature_params=params['features'],
                              dataset_evaluation_mode=dataset_evaluation_mode,
                              classifier_method=params['classifier']['method'],
                              overwrite=params['general']['overwrite']
                              )

            foot()

        # System evaluation
        # ==================================================
        if params['flow']['evaluate_system']:
            section_header('System evaluation')

            do_system_evaluation(dataset=dataset,
                                 dataset_evaluation_mode=dataset_evaluation_mode,
                                 result_path=params['path']['results'])

            foot()

    # System evaluation with challenge data
    elif not args.development and args.challenge:
        # Fetch data over internet and setup the data
        challenge_dataset = eval(params['general']['challenge_dataset'])()

        if params['flow']['initialize']:
            challenge_dataset.fetch()

        # System testing
        if params['flow']['test_system']:
            section_header('System testing with challenge data')

            do_system_testing(dataset=challenge_dataset,
                              feature_path=params['path']['features'],
                              result_path=params['path']['challenge_results'],
                              model_path=params['path']['models'],
                              feature_params=params['features'],
                              dataset_evaluation_mode=dataset_evaluation_mode,
                              classifier_method=params['classifier']['method'],
                              overwrite=True
                              )

            foot()

            print " "
            print "Your results for the challenge data are stored at ["+params['path']['challenge_results']+"]"
            print " "
    return 0


def process_parameters(params):
    """Parameter post-processing.

    Parameters
    ----------
    params : dict
        parameters in dict

    Returns
    -------
    params : dict
        processed parameters

    """

    # Convert feature extraction window and hop sizes seconds to samples
    params['features']['mfcc']['win_length'] = int(params['features']['win_length_seconds'] * params['features']['fs'])
    params['features']['mfcc']['hop_length'] = int(params['features']['hop_length_seconds'] * params['features']['fs'])

    # Copy parameters for current classifier method
    params['classifier']['parameters'] = params['classifier_parameters'][params['classifier']['method']]

    # Hash
    params['features']['hash'] = get_parameter_hash(params['features'])
    params['classifier']['hash'] = get_parameter_hash(params['classifier'])

    # Paths
    params['path']['data'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['data'])
    params['path']['base'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['base'])

    # Features
    params['path']['features_'] = params['path']['features']
    params['path']['features'] = os.path.join(params['path']['base'],
                                              params['path']['features'],
                                              params['features']['hash'])

    # Feature normalizers
    params['path']['feature_normalizers_'] = params['path']['feature_normalizers']
    params['path']['feature_normalizers'] = os.path.join(params['path']['base'],
                                                         params['path']['feature_normalizers'],
                                                         params['features']['hash'])

    # Models
    params['path']['models_'] = params['path']['models']
    params['path']['models'] = os.path.join(params['path']['base'],
                                            params['path']['models'],
                                            params['features']['hash'], params['classifier']['hash'])
    # Results
    params['path']['results_'] = params['path']['results']
    params['path']['results'] = os.path.join(params['path']['base'],
                                             params['path']['results'],
                                             params['features']['hash'], params['classifier']['hash'])

    return params


def make_folders(params, parameter_filename='parameters.yaml'):
    """Create all needed folders, and saves parameters in yaml-file for easier manual browsing of data.

    Parameters
    ----------
    params : dict
        parameters in dict

    parameter_filename : str
        filename to save parameters used to generate the folder name

    Returns
    -------
    nothing

    """

    # Check that target path exists, create if not
    check_path(params['path']['features'])
    check_path(params['path']['feature_normalizers'])
    check_path(params['path']['models'])
    check_path(params['path']['results'])

    # Save parameters into folders to help manual browsing of files.

    # Features
    feature_parameter_filename = os.path.join(params['path']['features'], parameter_filename)
    if not os.path.isfile(feature_parameter_filename):
        save_parameters(feature_parameter_filename, params['features'])

    # Feature normalizers
    feature_normalizer_parameter_filename = os.path.join(params['path']['feature_normalizers'], parameter_filename)
    if not os.path.isfile(feature_normalizer_parameter_filename):
        save_parameters(feature_normalizer_parameter_filename, params['features'])

    # Models
    model_features_parameter_filename = os.path.join(params['path']['base'],
                                                     params['path']['models_'],
                                                     params['features']['hash'],
                                                     parameter_filename)
    if not os.path.isfile(model_features_parameter_filename):
        save_parameters(model_features_parameter_filename, params['features'])

    model_models_parameter_filename = os.path.join(params['path']['base'],
                                                   params['path']['models_'],
                                                   params['features']['hash'],
                                                   params['classifier']['hash'],
                                                   parameter_filename)
    if not os.path.isfile(model_models_parameter_filename):
        save_parameters(model_models_parameter_filename, params['classifier'])

    # Results
    # Save parameters into folders to help manual browsing of files.
    result_features_parameter_filename = os.path.join(params['path']['base'],
                                                      params['path']['results_'],
                                                      params['features']['hash'],
                                                      parameter_filename)
    if not os.path.isfile(result_features_parameter_filename):
        save_parameters(result_features_parameter_filename, params['features'])

    result_models_parameter_filename = os.path.join(params['path']['base'],
                                                    params['path']['results_'],
                                                    params['features']['hash'],
                                                    params['classifier']['hash'],
                                                    parameter_filename)
    if not os.path.isfile(result_models_parameter_filename):
        save_parameters(result_models_parameter_filename, params['classifier'])


def get_feature_filename(audio_file, path, extension='cpickle'):
    """Get feature filename

    Parameters
    ----------
    audio_file : str
        audio file name from which the features are extracted

    path :  str
        feature path

    extension : str
        file extension
        (Default value='cpickle')

    Returns
    -------
    feature_filename : str
        full feature filename

    """

    audio_filename = os.path.split(audio_file)[1]
    return os.path.join(path, os.path.splitext(audio_filename)[0] + '.' + extension)


def get_feature_normalizer_filename(fold, path, extension='cpickle'):
    """Get normalizer filename

    Parameters
    ----------
    fold : int >= 0
        evaluation fold number

    path :  str
        normalizer path

    extension : str
        file extension
        (Default value='cpickle')

    Returns
    -------
    normalizer_filename : str
        full normalizer filename

    """

    return os.path.join(path, 'scale_fold' + str(fold) + '.' + extension)


def get_model_filename(fold, path, extension='cpickle'):
    """Get model filename

    Parameters
    ----------
    fold : int >= 0
        evaluation fold number

    path :  str
        model path

    extension : str
        file extension
        (Default value='cpickle')

    Returns
    -------
    model_filename : str
        full model filename

    """

    return os.path.join(path, 'model_fold' + str(fold) + '.' + extension)


def get_result_filename(fold, path, extension='txt'):
    """Get result filename

    Parameters
    ----------
    fold : int >= 0
        evaluation fold number

    path :  str
        result path

    extension : str
        file extension
        (Default value='cpickle')

    Returns
    -------
    result_filename : str
        full result filename

    """

    if fold == 0:
        return os.path.join(path, 'results.' + extension)
    else:
        return os.path.join(path, 'results_fold' + str(fold) + '.' + extension)


def do_feature_extraction(files, dataset, feature_path, params, overwrite=False):
    """Feature extraction

    Parameters
    ----------
    files : list
        file list

    dataset : class
        dataset class

    feature_path : str
        path where the features are saved

    params : dict
        parameter dict

    overwrite : bool
        overwrite existing feature files
        (Default value=False)

    Returns
    -------
    nothing

    Raises
    -------
    IOError
        Audio file not found.

    """

    # Check that target path exists, create if not
    check_path(feature_path)
    Parallel(n_jobs=8)(delayed(run_job)(file_id, audio_filename, files, dataset, feature_path, params, overwrite) for file_id, audio_filename in enumerate(files))
    # for file_id, audio_filename in enumerate(files):
    #     run_job(file_id, audio_filename, files, dataset, feature_path, params, overwrite)


def run_job(file_id, audio_filename, files, dataset, feature_path, params, overwrite):
    # Get feature filename
    current_feature_file = get_feature_filename(audio_file=os.path.split(audio_filename)[1], path=feature_path)

    progress(title_text='Extracting',
             percentage=(float(file_id) / len(files)),
             note=os.path.split(audio_filename)[1])

    if not os.path.isfile(current_feature_file) or overwrite:
        # Load audio data
        if os.path.isfile(dataset.relative_to_absolute_path(audio_filename)):
            y, fs = load_audio(filename=dataset.relative_to_absolute_path(audio_filename), mono=True, fs=params['fs'])
        else:
            raise IOError("Audio file not found [%s]" % audio_filename)

        # Extract features
        extended = extend_dataset(y, fs)
        feature_data = [feature_extraction(y=y_ex,
                               fs=fs,
                               include_mfcc0=params['include_mfcc0'],
                               include_delta=params['include_delta'],
                               include_acceleration=params['include_acceleration'],
                               mfcc_params=params['mfcc'],
                               delta_params=params['mfcc_delta'],
                               acceleration_params=params['mfcc_acceleration']) for y_ex in extended]
        # Save
        save_data(current_feature_file, feature_data)


def do_feature_normalization(dataset, feature_normalizer_path, feature_path, dataset_evaluation_mode='folds', overwrite=False):
    """Feature normalization

    Calculated normalization factors for each evaluation fold based on the training material available.

    Parameters
    ----------
    dataset : class
        dataset class

    feature_normalizer_path : str
        path where the feature normalizers are saved.

    feature_path : str
        path where the features are saved.

    dataset_evaluation_mode : str ['folds', 'full']
        evaluation mode, 'full' all material available is considered to belong to one fold.
        (Default value='folds')

    overwrite : bool
        overwrite existing normalizers
        (Default value=False)

    Returns
    -------
    nothing

    Raises
    -------
    IOError
        Feature file not found.

    """

    # Check that target path exists, create if not
    check_path(feature_normalizer_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_normalizer_file = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)

        if not os.path.isfile(current_normalizer_file) or overwrite:
            # Initialize statistics
            file_count = len(dataset.train(fold))
            normalizer = FeatureNormalizer()

            for item_id, item in enumerate(dataset.train(fold)):
                progress(title_text='Collecting data',
                         fold=fold,
                         percentage=(float(item_id) / file_count),
                         note=os.path.split(item['file'])[1])
                # Load features
                if os.path.isfile(get_feature_filename(audio_file=item['file'], path=feature_path)):
                    feature_list = load_data(get_feature_filename(audio_file=item['file'], path=feature_path))
                    feature_data = [x['stat'] for x in feature_list]
                else:
                    raise IOError("Feature file not found [%s]" % (item['file']))

                # Accumulate statistics
                for f in feature_data:
                    normalizer.accumulate(f)

            # Calculate normalization factors
            normalizer.finalize()

            # Save
            save_data(current_normalizer_file, normalizer)


def do_system_training(dataset, model_path, feature_normalizer_path, feature_path, classifier_params,
                       dataset_evaluation_mode='folds', classifier_method='gmm', overwrite=False, feature_params=None):
    """System training

    model container format:

    {
        'normalizer': normalizer class
        'models' :
            {
                'office' : mixture.GMM class
                'home' : mixture.GMM class
                ...
            }
    }

    Parameters
    ----------
    dataset : class
        dataset class

    model_path : str
        path where the models are saved.

    feature_normalizer_path : str
        path where the feature normalizers are saved.

    feature_path : str
        path where the features are saved.

    classifier_params : dict
        parameter dict

    dataset_evaluation_mode : str ['folds', 'full']
        evaluation mode, 'full' all material available is considered to belong to one fold.
        (Default value='folds')

    classifier_method : str ['gmm', 'rnn']
        classifier method, currently only GMM and RNN supported
        (Default value='gmm')

    overwrite : bool
        overwrite existing models
        (Default value=False)

    Returns
    -------
    nothing

    Raises
    -------
    ValueError
        classifier_method is unknown.

    IOError
        Feature normalizer not found.
        Feature file not found.

    """

    if classifier_method not in {'rnn'}:
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(model_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_model_file = get_model_filename(fold=fold, path=model_path)
        if not os.path.isfile(current_model_file) or overwrite:
            # Load normalizer
            feature_normalizer_filename = get_feature_normalizer_filename(fold=fold, path=feature_normalizer_path)
            if os.path.isfile(feature_normalizer_filename):
                normalizer = load_data(feature_normalizer_filename)
            else:
                raise IOError("Feature normalizer not found [%s]" % feature_normalizer_filename)

            # Initialize model container
            model_container = {'normalizer': normalizer, 'models': {}}

            # Collect labels
            data_target = []
            all_targets = {}
            for item in dataset.train(fold):
                if item['scene_label'] not in all_targets:
                    all_targets[item['scene_label']] = len(all_targets)
                data_target.append(all_targets[item['scene_label']])
            data_labels = numpy.array(data_target).reshape(-1,1)
            onehot = preprocessing.OneHotEncoder()
            onehot.fit(data_labels)

            # Collect training examples
            def super_yielder():
                file_count = len(dataset.train(fold))
                while True:
                    for item_id, item in enumerate(dataset.train(fold)):
                        # Load features
                        feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)
                        if os.path.isfile(feature_filename):
                            feature_data = [x['feat'] for x in load_data(feature_filename)]
                        else:
                            raise IOError("Features not found [%s]" % (item['file']))

                        # Scale features
                        feature_data = [model_container['normalizer'].normalize(f) for f in feature_data]

                        data = {}
                        data_feat = []
                        data_target = []
                        for f in feature_data:
                            f = f[:2000] # TODO: this is a temporary fix for non-equal input features
                            data_feat.append(f[numpy.newaxis])
                            data_target.append(all_targets[item['scene_label']])
                        data_feat = numpy.vstack(data_feat)
                        data_labels = numpy.array(data_target).reshape(-1,1)
                        targets = onehot.transform(data_labels)
                        yield data_feat, targets.toarray()

            # Train models
            if classifier_method == 'rnn':
                # order = numpy.random.permutation(len(data_labels))
                # data_feat = data_feat[order,:,:]
                # targets = targets[order,:]
                """
                CAREFUL!! This is only for graphing purposes
                """
                test_data, test_labels = collect_test_data(dataset, fold, normalizer, feature_path, feature_params)
                test_labels = [all_targets[lab] for lab in test_labels]
                test_target = onehot.transform(numpy.array(test_labels).reshape(-1,1)).toarray()
                """
                """
                model_container['models']['model'] = RNN(**classifier_params)
                model_container['models']['model'].fit(super_yielder(),
                                                       validation_data=(test_data,test_target),
                                                       samples=len(dataset.train(fold)))
                # model_container['models']['model'].fit(data_feat, targets, validation_data=(test_data,test_target))
                model_container['models']['model'].set_filename(current_model_file + '_weights.h5')
                model_container['models']['labels'] = {v: k for k, v in all_targets.items()}
            else:
                raise ValueError("Unknown classifier method ["+classifier_method+"]")

            # Save models
            save_data(current_model_file, model_container)


def do_system_testing(dataset, result_path, feature_path, model_path, feature_params,
                      dataset_evaluation_mode='folds', classifier_method='rnn', overwrite=False):
    """System testing.

    If extracted features are not found from disk, they are extracted but not saved.

    Parameters
    ----------
    dataset : class
        dataset class

    result_path : str
        path where the results are saved.

    feature_path : str
        path where the features are saved.

    model_path : str
        path where the models are saved.

    feature_params : dict
        parameter dict

    dataset_evaluation_mode : str ['folds', 'full']
        evaluation mode, 'full' all material available is considered to belong to one fold.
        (Default value='folds')

    classifier_method : str ['gmm', 'rnn']
        classifier method, currently only GMM and RNN supported
        (Default value='gmm')

    overwrite : bool
        overwrite existing models
        (Default value=False)

    Returns
    -------
    nothing

    Raises
    -------
    ValueError
        classifier_method is unknown.

    IOError
        Model file not found.
        Audio file not found.

    """

    if not classifier_method in {'rnn'}:
        raise ValueError("Unknown classifier method ["+classifier_method+"]")

    # Check that target path exists, create if not
    check_path(result_path)

    for fold in dataset.folds(mode=dataset_evaluation_mode):
        current_result_file = get_result_filename(fold=fold, path=result_path)
        if not os.path.isfile(current_result_file) or overwrite:
            results = []

            # Load class model container
            model_filename = get_model_filename(fold=fold, path=model_path)
            if os.path.isfile(model_filename):
                model_container = load_data(model_filename)
            else:
                raise IOError("Model file not found [%s]" % model_filename)

            file_count = len(dataset.test(fold))
            for file_id, item in enumerate(dataset.test(fold)):
                progress(title_text='Testing',
                         fold=fold,
                         percentage=(float(file_id) / file_count),
                         note=os.path.split(item['file'])[1])

                # Load features
                feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)

                if os.path.isfile(feature_filename):
                    feature_data = load_data(feature_filename)['feat']
                else:
                    # Load audio
                    audio_filename = dataset.relative_to_absolute_path(item['file'])
                    if os.path.isfile(audio_filename):
                        y, fs = load_audio(filename=audio_filename, mono=True, fs=feature_params['fs'])
                    else:
                        raise IOError("Audio file not found [%s]" % (item['file']))

                    feature_data = feature_extraction(y=y,
                                                      fs=fs,
                                                      include_mfcc0=feature_params['include_mfcc0'],
                                                      include_delta=feature_params['include_delta'],
                                                      include_acceleration=feature_params['include_acceleration'],
                                                      mfcc_params=feature_params['mfcc'],
                                                      delta_params=feature_params['mfcc_delta'],
                                                      acceleration_params=feature_params['mfcc_acceleration'],
                                                      statistics=False)['feat']

                # Normalize features
                feature_data = model_container['normalizer'].normalize(feature_data)

                # Do classification for the block
                if classifier_method == 'rnn':
                    class_idx = model_container['models']['model'].predict(feature_data[numpy.newaxis])
                    current_result = model_container['models']['labels'][class_idx]
                else:
                    raise ValueError("Unknown classifier method ["+classifier_method+"]")

                # Store the result
                results.append((dataset.absolute_to_relative(item['file']), current_result))

            # Save testing results
            with open(current_result_file, 'wt') as f:
                writer = csv.writer(f, delimiter='\t')
                for result_item in results:
                    writer.writerow(result_item)


def collect_test_data(dataset, fold, normalizer, feature_path, feature_params):

    file_count = len(dataset.test(fold))
    features = []
    y_true = []
    for file_id, item in enumerate(dataset.test(fold)):
        # Load features
        feature_filename = get_feature_filename(audio_file=item['file'], path=feature_path)

        if os.path.isfile(feature_filename):
            feature_data = load_data(feature_filename)
            if type(feature_data) == list:
                feature_data = feature_data[0]
            feature_data = feature_data['feat']
        else:
            # Load audio
            audio_filename = dataset.relative_to_absolute_path(item['file'])
            if os.path.isfile(audio_filename):
                y, fs = load_audio(filename=audio_filename, mono=True, fs=feature_params['fs'])
            else:
                raise IOError("Audio file not found [%s]" % (item['file']))

            feature_data = feature_extraction(y=y,
                                              fs=fs,
                                              include_mfcc0=feature_params['include_mfcc0'],
                                              include_delta=feature_params['include_delta'],
                                              include_acceleration=feature_params['include_acceleration'],
                                              mfcc_params=feature_params['mfcc'],
                                              delta_params=feature_params['mfcc_delta'],
                                              acceleration_params=feature_params['mfcc_acceleration'],
                                              statistics=False)['feat']

        # Normalize features
        feature_data = normalizer.normalize(feature_data)
        features.append(feature_data[numpy.newaxis])

        y_true.append(dataset.file_meta(item['file'])[0]['scene_label'])

    features = numpy.vstack(features)
    return features, y_true


def do_system_evaluation(dataset, result_path, dataset_evaluation_mode='folds'):
    """System evaluation. Testing outputs are collected and evaluated. Evaluation results are printed.

    Parameters
    ----------
    dataset : class
        dataset class

    result_path : str
        path where the results are saved.

    dataset_evaluation_mode : str ['folds', 'full']
        evaluation mode, 'full' all material available is considered to belong to one fold.
        (Default value='folds')

    Returns
    -------
    nothing

    Raises
    -------
    IOError
        Result file not found

    """

    dcase2016_scene_metric = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
    results_fold = []
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        dcase2016_scene_metric_fold = DCASE2016_SceneClassification_Metrics(class_list=dataset.scene_labels)
        results = []
        result_filename = get_result_filename(fold=fold, path=result_path)

        if os.path.isfile(result_filename):
            with open(result_filename, 'rt') as f:
                for row in csv.reader(f, delimiter='\t'):
                    results.append(row)
        else:
            raise IOError("Result file not found [%s]" % result_filename)

        y_true = []
        y_pred = []
        for result in results:
            y_true.append(dataset.file_meta(result[0])[0]['scene_label'])
            y_pred.append(result[1])
        dcase2016_scene_metric.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
        dcase2016_scene_metric_fold.evaluate(system_output=y_pred, annotated_ground_truth=y_true)
        results_fold.append(dcase2016_scene_metric_fold.results())
    results = dcase2016_scene_metric.results()

    print "  File-wise evaluation, over %d folds" % dataset.fold_count
    fold_labels = ''
    separator = '     =====================+======+======+==========+  +'
    if dataset.fold_count > 1:
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            fold_labels += " {:8s} |".format('Fold'+str(fold))
            separator += "==========+"
    print "     {:20s} | {:4s} : {:4s} | {:8s} |  |".format('Scene label', 'Nref', 'Nsys', 'Accuracy')+fold_labels
    print separator
    for label_id, label in enumerate(sorted(results['class_wise_accuracy'])):
        fold_values = ''
        if dataset.fold_count > 1:
            for fold in dataset.folds(mode=dataset_evaluation_mode):
                fold_values += " {:5.1f} %  |".format(results_fold[fold-1]['class_wise_accuracy'][label] * 100)
        print "     {:20s} | {:4d} : {:4d} | {:5.1f} %  |  |".format(label,
                                                                     results['class_wise_data'][label]['Nref'],
                                                                     results['class_wise_data'][label]['Nsys'],
                                                                     results['class_wise_accuracy'][label] * 100)+fold_values
    print separator
    fold_values = ''
    if dataset.fold_count > 1:
        for fold in dataset.folds(mode=dataset_evaluation_mode):
            fold_values += " {:5.1f} %  |".format(results_fold[fold-1]['overall_accuracy'] * 100)

    print "     {:20s} | {:4d} : {:4d} | {:5.1f} %  |  |".format('Overall accuracy',
                                                                 results['Nref'],
                                                                 results['Nsys'],
                                                                 results['overall_accuracy'] * 100)+fold_values

if __name__ == "__main__":
    #try:
        sys.exit(main(sys.argv))
    #except (ValueError, IOError) as e:
    #    sys.exit(e)
