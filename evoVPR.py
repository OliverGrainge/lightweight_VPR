from QuantizationSim import Quantizer 
from torchvision.models import resnet18
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn as nn 
from model import network
import util
import datasets_ws
import parser
import numpy as np
import test
import torch.nn.functional as F
import copy
import time 
import os

args = parser.parse_arguments()

args.datasets_folder = "/home/oliver/Documents/github/lightweight_VPR/datasets_vg/datasets"
args.dataset_name = "pitts30k"
args.infer_batch_size=3
args.device = 'cuda:0'
args.resume = f"/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_{args.aggregation}_{str(args.fc_output_dim)}/best_model.pth"
model = network.GeoLocalizationNet(args).eval().to(args.device)
model = util.resume_model(args, model)
test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")


NUM_SAMPLES = 50
BATCH_SIZE = 50
VALUES = {'int4': 4, 'int8': 8, 'fp16': 16}
MAX_GENERATIONS = 5000
POPULATION_SIZE = 16
NUM_LAYERS = 103
AVERAGE_BITWIDTH = args.average_bitwidth
START_MUTATION_RATE = 0.2
MUTATION_RATE = START_MUTATION_RATE
END_MUTATION_RATE = 0.5
MUTATION_DELTA = (START_MUTATION_RATE - END_MUTATION_RATE) / 2000
SAMPLE_SIZE = 8


# Generate random indices
indices = torch.randperm(len(test_ds))[:NUM_SAMPLES]

# Create a subset
test_ds = Subset(test_ds, indices)
test_dl = DataLoader(test_ds, BATCH_SIZE)
for batch in test_dl:
    break

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
quantizer = Quantizer(model,
        layer_precision="int8",
        activation_precision="fp32",
        activation_granularity="tensor",
        layer_granularity="channel",
        calibration_type="minmax", 
        cal_samples=10,
        calibration_loader=test_dl,
        activation_layers = (nn.ReLU, nn.ReLU6), 
        device=args.device, 
        number_of_activations=32,
        number_of_layers=103)

quantizer.quantize_layers()

all_layer_types = quantizer.all_layer_types

def config_length(layer_types):
    idx = 0
    for i in range(len(layer_types)):
        if layer_types[i].startswith('Conv2d') and layer_types[i+1].startswith('BatchNorm2d'):
            idx += 1
        elif layer_types[i] == "Linear":
            idx += 1
        elif layer_types[i].startswith('Conv2d') and layer_types[i + 1].startswith('Conv2d'):
            idx += 1
    return idx


NUM_LAYERS = config_length(quantizer.all_layer_types)
        
def convert_config(config, layer_types):
    new_config = []
    i = 0
    idx = 0
    idx = 0
    while len(new_config) < len(layer_types):
        if layer_types[idx].startswith('Conv2d') and layer_types[idx+1].startswith('BatchNorm2d'):
            new_config.append(config[i])
            new_config.append(config[i])
            idx += 2
            i += 1
        elif layer_types[idx] == "Linear":
            new_config.append(config[i])
            idx += 1
            i += 1
        elif layer_types[idx].startswith('Conv2d') and layer_types[idx + 1].startswith('Conv2d'):
            new_config.append(config[i])
            idx += 1
            i += 1
    return new_config

def init_pop_old(popsize: int=100) -> list:
    population = []
    while len(population) < popsize:
        ind = np.random.choice(["int4", "int8", "fp16"], size=NUM_LAYERS)
        if np.mean([VALUES[gene] for gene in ind]) <= AVERAGE_BITWIDTH:
            population.append(ind)
    return population

def init_pop(popsize: int=100) -> list:
    base_ind = np.array(["fp16" for _ in range(10)] + ["int8" for _ in range(20)] + ["int4" for _ in range(20)] + ['fp16', 'fp16'])
    probs = [np.array([0.33, 0.33, 0.34]) for _ in range(len(base_ind))]
    population = []
    while len(population) < popsize:
        ind = offspring(base_ind, probs=probs)
        if np.mean([VALUES[gene] for gene in ind]) < AVERAGE_BITWIDTH:
            population.append(ind)
    return population


def offspring(individual: np.array, probs: list) -> np.array:
    individual = copy.deepcopy(individual)
    child = []
    for i in range(len(individual)):
        if np.random.rand() < MUTATION_RATE:
            new_gene = np.random.choice(np.array(["int4", "int8", "fp16"]), p=probs[i])
            child.append(new_gene)
        else:
            child.append(individual[i])
    return np.array(child)


def cost(q_features: torch.tensor, target_features: torch.tensor) -> float:
    return F.mse_loss(q_features, target_features)
    

def compute_sensitivity():
    configuration = ['fp32' for _ in range(NUM_LAYERS)]

    costs = []
    with torch.no_grad():
        desc = model(batch[0].to(args.device)).detach().cpu()
        #desc = torch.stack([model(batch[0].to(args.device)).detach().cpu() for batch in test_dl])

    for i in range(len(configuration)):
        config = configuration 
        config[i] = 'int4'
        long_config = convert_config(config, all_layer_types)
        quantizer = Quantizer(model,
                layer_configuration=long_config,
                activation_precision="fp32",
                activation_granularity="tensor",
                layer_granularity="channel",
                calibration_type="minmax", 
                cal_samples=10,
                calibration_loader=test_dl,
                activation_layers = (nn.ReLU, nn.ReLU6), 
                device=args.device, 
                number_of_activations=32,
                number_of_layers=103)
        
        qmodel = quantizer.quantize_layers()

        with torch.no_grad():
            qdesc = qmodel(batch[0].to(args.device)).detach().cpu()
            #qdesc = torch.stack([qmodel(batch[0].to(args.device)).detach().cpu() for batch in test_dl])
        costs.append(cost(qdesc, desc).item())
    costs = (costs - np.min(costs))/(np.max(costs) - np.min(costs))
    costs = list(costs)
    base_dist = np.array([0.33, 0.33, 0.34])
    tau = 0.15
    probs = [base_dist + tau * c * np.array([-1., 0., 1.]) for c in costs]
    for prob in probs:
        assert np.sum(prob) == 1.
        assert np.all(prob >= 0)
    return probs



def crossover(ind1: list, ind2: list):
    idx_min = int(0.1 * len(ind1))
    idx_max = int(0.9 * len(ind1))

    idx = np.random.randint(idx_min, idx_max)
    
    if np.random.rand() > 0.5:
        child = np.concatenate((ind1[:idx], ind2[idx:]))
    else: 
        child = np.concatenate((ind1[:idx], ind2[idx:]))
    assert len(child) == len(ind2)
    return child

def selection(population: list) -> list: 
    # returns best parent 
    best_individual = None 
    worst_individual = None
    best_fitness = np.inf 
    worst_fitness = -np.inf 

    model = network.GeoLocalizationNet(args).eval()
    model = util.resume_model(args, model)
    model.to(args.device).eval()
    with torch.no_grad():
        desc = model(batch[0].to(args.device)).detach().cpu()
        #desc = torch.stack([model(batch[0].to(args.device)).detach().cpu() for batch in test_dl])
    fitness_scores = []
    for i, individual in enumerate(population):
        print("Assesing Individual", i)
        long_individual = convert_config(individual, all_layer_types)
        quantizer = Quantizer(model,
            layer_configuration=long_individual,
            activation_precision="fp32",
            activation_granularity="tensor",
            layer_granularity="channel",
            calibration_type="minmax", 
            cal_samples=10,
            calibration_loader=test_dl,
            activation_layers = (nn.ReLU, nn.ReLU6), 
            device=args.device, 
            number_of_activations=32,
            number_of_layers=103)
        qmodel = quantizer.quantize_layers()
        qmodel = qmodel.eval()

        with torch.no_grad():
            qdesc = qmodel(batch[0].to(args.device)).detach().cpu()
            #qdesc = torch.stack([qmodel(batch[0].to(args.device)).detach().cpu() for batch in test_dl])

        fitness = cost(qdesc, desc)
        fitness_scores.append(fitness.item())

        if fitness < best_fitness: 
            best_fitness = fitness
            best_individual = individual
            best_idx = i
        
        if fitness > worst_fitness:
            worst_fitness = fitness
            worst_individual = individual
            worst_idx = i
    print(np.argsort(np.array(fitness_scores)[0]))
    second_best_individual = population[int(np.argsort(np.array(fitness_scores)[0]))]
    return best_individual, second_best_individual, worst_individual, best_fitness, worst_fitness, best_idx, worst_idx, fitness_scores

probs = compute_sensitivity()
all_fitness = []
population = np.array(init_pop(popsize=POPULATION_SIZE))
all_fitness = []
old_best_idx = None
old_population = population
for generation in range(MAX_GENERATIONS):
    st = time.time()
    old_population = copy.deepcopy(population)
    random_indices = np.random.choice(len(population), SAMPLE_SIZE, replace=False)
    sample_population = population[random_indices]
    best_individual, second_best_individual, worst_individual, best_fitness, worst_fitness, best_idx, worst_idx, fitness_scores = selection(sample_population)
    all_fitness.append(best_fitness)

    child = crossover(best_individual, second_best_individual)
    average_bitwidth = np.inf
    while average_bitwidth > AVERAGE_BITWIDTH:
        child = offspring(child, probs)
        average_bitwidth = np.mean([VALUES[gene] for gene in child])
    sample_population[worst_idx] = child

    population[random_indices] = sample_population


    if not os.path.exists(args.save_dir + "/" + args.backbone + "_" + args.aggregation + "_" + str(args.fc_output_dim) + "_" + str(AVERAGE_BITWIDTH)):
        os.makedirs(args.save_dir + "/" + args.backbone + "_" + args.aggregation + "_" + str(args.fc_output_dim) + "_" + str(AVERAGE_BITWIDTH))
    np.save(args.save_dir + "/" + args.backbone + "_" + args.aggregation + "_" + str(args.fc_output_dim) + "_" + str(AVERAGE_BITWIDTH) + "/all_fitness.npy", np.array(all_fitness))
    np.save(args.save_dir + "/" + args.backbone + "_" + args.aggregation + "_" + str(args.fc_output_dim) + "_" + str(AVERAGE_BITWIDTH) + "/best_configuration.npy", best_individual)
    et = time.time()

    if MUTATION_RATE > END_MUTATION_RATE:
        MUTATION_RATE = MUTATION_RATE - MUTATION_DELTA

    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print("###################################################################################################3")
    print(" ")
    print("############################### GENERATION ", generation, " FITNESS ", best_fitness.item(), "#################################")
    print(" ")
    print(" ")
    print("============================", [(population[i] == old_population[i]).all() for i in range(len(population))])
    print(" ")
    print("------------------------------------- GENERATION TOOK: ", et - st, "seconds")
    print('========== MUTATION_RATE: ', MUTATION_RATE)
    print(" ")
    print("###################################################################################################3")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
