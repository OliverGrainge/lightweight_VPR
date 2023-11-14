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

args = parser.parse_arguments()

args.aggregation = "mac"
args.datasets_folder = "/home/oliver/Documents/github/lightweight_VPR/datasets_vg/datasets"
args.backbone = "mobilenetv2conv4"
args.dataset_name = "pitts30k"
args.fc_output_dim = 1024
args.infer_batch_size=3
args.device = 'cuda:0'
args.resume = f"/home/oliver/Documents/github/lightweight_VPR/weights/fp32/mobilenetv2conv4_{args.aggregation}_{str(args.fc_output_dim)}/best_model.pth"
model = network.GeoLocalizationNet(args).eval().to(args.device)
model = util.resume_model(args, model)

test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")


NUM_SAMPLES = 50
BATCH_SIZE = 50
VALUES = {'int4': 4, 'int8': 8, 'fp16': 16}
MAX_GENERATIONS = 1000
POPULATION_SIZE = 50
NUM_LAYERS = 102
AVERAGE_BITWIDTH = 9
START_MUTATION_RATE = 0.2
MUTATION_RATE = START_MUTATION_RATE
END_MUTATION_RATE = 0.01
MUTATION_DELTA = (START_MUTATION_RATE - END_MUTATION_RATE) / 600


# Generate random indices
indices = torch.randperm(len(test_ds))[:NUM_SAMPLES]

# Create a subset
test_ds = Subset(test_ds, indices)
test_dl = DataLoader(test_ds, BATCH_SIZE)
for batch in test_dl:
    break

def init_pop(popsize: int=100) -> list:
    cal_dl = DataLoader(test_ds, batch_size=4)

    quantizer = Quantizer(model,
        layer_precision="fp32",
        activation_precision="fp32",
        activation_granularity="tensor",
        layer_granularity="tensor",
        calibration_type="minmax", 
        cal_samples=10,
        calibration_loader=cal_dl,
        activation_layers = (nn.ReLU, nn.ReLU6), 
        device=args.device)

    num_layers = quantizer.num_layers()

    population = []
    idx = 0
    while idx < popsize:
        individual = np.random.choice(['int4', 'int8', 'fp16'], size=num_layers)
        if np.mean([VALUES[gene] for gene in individual]) <= AVERAGE_BITWIDTH:
            population.append(individual)
            idx += 1
    return population

out = init_pop(10)

def offspring(individual: np.array, probs: list) -> np.array:
    individual = copy.deepcopy(individual)
    child = []
    for i in range(len(individual)):
        if np.random.rand() > MUTATION_RATE:
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
        quantizer = Quantizer(model,
                layer_configuration=config,
                activation_precision="fp32",
                activation_granularity="tensor",
                layer_granularity="channel",
                calibration_type="minmax", 
                cal_samples=10,
                calibration_loader=test_dl,
                activation_layers = (nn.ReLU, nn.ReLU6), 
                device=args.device, 
                number_of_activations=32,
                number_of_layers=102)
        
        qmodel = quantizer.quantize_layers()

        with torch.no_grad():
            qdesc = qmodel(batch[0].to(args.device)).detach().cpu()
            #qdesc = torch.stack([qmodel(batch[0].to(args.device)).detach().cpu() for batch in test_dl])
        costs.append(cost(qdesc, desc).item())
    costs = (costs - np.min(costs))/(np.max(costs) - np.min(costs))
    costs = list(costs)
    base_dist = np.array([0.33, 0.33, 0.34])
    tau = 0.1
    probs = [base_dist + tau * c * np.array([-1., 0., 1.]) for c in costs]
    for prob in probs:
        assert np.sum(prob) == 1.
        assert np.all(prob >= 0)
    return probs





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
        quantizer = Quantizer(model,
            layer_configuration=individual,
            activation_precision="fp32",
            activation_granularity="tensor",
            layer_granularity="channel",
            calibration_type="minmax", 
            cal_samples=10,
            calibration_loader=test_dl,
            activation_layers = (nn.ReLU, nn.ReLU6), 
            device=args.device, 
            number_of_activations=32,
            number_of_layers=102)
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

    return best_individual, worst_individual, best_fitness, worst_fitness, best_idx, worst_idx, fitness_scores

probs = compute_sensitivity()
all_fitness = []
population = init_pop(popsize=POPULATION_SIZE)
all_fitness = []
old_best_idx = None
old_population = population
for generation in range(MAX_GENERATIONS):
    st = time.time()
    old_population = copy.deepcopy(population)
    best_individual, worst_individual, best_fitness, worst_fitness, best_idx, worst_idx, fitness_scores = selection(population)
    all_fitness.append(best_fitness)



    parent = best_individual
    average_bitwidth = np.inf
    while average_bitwidth > AVERAGE_BITWIDTH:
        child = offspring(parent, probs)
        average_bitwidth = np.mean([VALUES[gene] for gene in child])
    population[worst_idx] = child
    np.save("all_fitness.npy", np.array(all_fitness))
    np.save("best_configuration.npy", best_individual)
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
