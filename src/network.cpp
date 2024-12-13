#include "include/network.hpp"

// Neural Network implementation using PyTorch
class NeuralNetwork : public torch::nn::Module {
private:
    torch::nn::Linear layer1, layer2, outputLayer;

public:
    NeuralNetwork(int stateSize, int hiddenSize, int actionSize)
        :   layer1(register_module("layer1", torch::nn::Linear(stateSize, hiddenSize))),
            layer2(register_module("layer2", torch::nn::Linear(hiddenSize, hiddenSize))),
            outputLayer(register_module("outputLayer", torch::nn::Linear(hiddenSize, actionSize)))
            {
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(layer1(x));
        x = torch::relu(layer2(x));
        x = outputLayer(x);
        return x;
    }
};

// Experience Replay Buffer
class ReplayBuffer {
private:
    std::deque<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> buffer;
    size_t maxSize;

public:
    ReplayBuffer(size_t size) : maxSize(size) {}

    // Adds a new experience (state, action, reward, next_state, done) to the buffer.
    void add(torch::Tensor state, torch::Tensor action, torch::Tensor reward,
             torch::Tensor nextState, torch::Tensor done) {
        if (buffer.size() >= maxSize) {
            buffer.pop_front();
        }
        buffer.push_back(std::make_tuple(state, action, reward, nextState, done));
    }

    // Randomly samples a batch of experiences from the buffer.
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
    sample(size_t batchSize) {
        std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> batch;
        std::random_device rd;
        std::mt19937 gen(rd());
        if(buffer.size() == 0) return batch;
        std::uniform_int_distribution<> dis(0, buffer.size() - 1);

        for (size_t i = 0; i < std::min(batchSize, buffer.size()); ++i) {
            batch.push_back(buffer[dis(gen)]);
        }
        return batch;
    }

    size_t size() const {
        return buffer.size();
    }
};

// Deep Q-Learning Agent
class DQLAgent {
private:
    torch::Device device;
    std::shared_ptr<NeuralNetwork> qNetwork;
    std::shared_ptr<NeuralNetwork> targetNetwork;
    ReplayBuffer replayBuffer;
    torch::optim::Adam optimizer;

    double gamma;
    double epsilon;
    double epsilonMin;
    double epsilonDecay;
    size_t stateSize;
    size_t actionSize;

public:
    DQLAgent(size_t stateSize, size_t actionSize, double gamma = 0.99,
             double epsilon = 1.0, double epsilonMin = 0.01, double epsilonDecay = 0.995, long long replayMemorySize = 10000)
        :   device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
            qNetwork(std::make_shared<NeuralNetwork>(stateSize, 64, actionSize)),
            targetNetwork(std::make_shared<NeuralNetwork>(stateSize, 64, actionSize)),
            replayBuffer(replayMemorySize),
            optimizer(qNetwork->parameters(), torch::optim::AdamOptions(1e-3)),
            gamma(gamma), epsilon(epsilon), epsilonMin(epsilonMin),
            epsilonDecay(epsilonDecay), stateSize(stateSize), actionSize(actionSize)
             {

        qNetwork->to(device);
        targetNetwork->to(device);
        syncTargetNetwork();
    }

    // Synchronizes the weights of the target network with those of the Q-network.
    void syncTargetNetwork(){
         targetNetwork->load_state_dict(qNetwork->state_dict());
    }

    // Selects an action based on the epsilon-greedy policy
    int getAction(const std::vector<double>& state) {
         torch::Tensor stateTensor = torch::tensor(state,  torch::dtype(torch::kFloat64)).to(device);

        qNetwork->eval();
        torch::NoGradGuard noGrad;
        torch::Tensor qValues = qNetwork->forward(stateTensor);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);

        if (dis(gen) < epsilon) {
            std::uniform_int_distribution<> actionDis(0, actionSize - 1);
            return actionDis(gen);
        }

        return qValues.argmax(0).item<int>();
    }

   void train() {
    if (replayBuffer.size() < 64) return;

    auto batch = replayBuffer.sample(64);

    // Unpack the batch into separate tensors
    std::vector<torch::Tensor> states, actions, rewards, next_states, dones;
    for (const auto& experience : batch) {
        states.push_back(std::get<0>(experience));
        actions.push_back(std::get<1>(experience));
        rewards.push_back(std::get<2>(experience));
        next_states.push_back(std::get<3>(experience));
        dones.push_back(std::get<4>(experience));
    }

    torch::Tensor stateBatch = torch::stack(states).to(device);
    torch::Tensor actionBatch = torch::stack(actions).to(device);
    torch::Tensor rewardBatch = torch::stack(rewards).to(device);
    torch::Tensor nextStateBatch = torch::stack(next_states).to(device);
    torch::Tensor doneBatch = torch::stack(dones).to(device);

    qNetwork->train();
    optimizer.zero_grad();

    // Get Q-values for the current states
    torch::Tensor currentQValues = qNetwork->forward(stateBatch).gather(1, actionBatch.unsqueeze(1));

    // Compute the target Q-values
    torch::NoGradGuard noGrad;
    torch::Tensor nextQValues = targetNetwork->forward(nextStateBatch).max(1).values;
    torch::Tensor targetQValues = rewardBatch + (1 - doneBatch) * gamma * nextQValues;

    // Compute loss (e.g., Mean Squared Error)
    torch::Tensor loss = torch::mse_loss(currentQValues.squeeze(1), targetQValues);

    // Backpropagate the error and update the weights
    loss.backward();
    optimizer.step();

    epsilon = std::max(epsilonMin, epsilon * epsilonDecay);
     static int updateCounter = 0;
    if (++updateCounter % 100 == 0) {
            syncTargetNetwork();
    }
}
    // Stores a new experience tuple in the replay buffer.
    void storeExperience(const std::vector<double>& state, int action, double reward,
                         const std::vector<double>& nextState, bool done) {
        torch::Tensor stateTensor = torch::tensor(state, torch::dtype(torch::kFloat64)).to(device);
        torch::Tensor actionTensor = torch::tensor({(long)action}).to(device);
        torch::Tensor rewardTensor = torch::tensor({reward}).to(device);
        torch::Tensor nextStateTensor = torch::tensor(nextState, torch::dtype(torch::kFloat64)).to(device);
        torch::Tensor doneTensor = torch::tensor({done}).to(device);

        replayBuffer.add(stateTensor, actionTensor, rewardTensor, nextStateTensor, doneTensor);
    }
};

class Environment {
public:
    std::vector<double> state;
    int stepCount;
    int maxSteps;

    Environment(size_t stateSize, int maxSteps = 200) : state(stateSize), stepCount(0), maxSteps(maxSteps) {
        reset();
    }

    // Reset environment to initial state
    void reset() {
        stepCount = 0;
        std::fill(state.begin(), state.end(), 0.0);
    }

    // Take an action and return (next_state, reward, done)
    std::tuple<std::vector<double>, double, bool> step(int action) {
        stepCount++;
        double reward = 0.0;
        if(action == 0){
            for(size_t i = 0; i < state.size(); ++i){
                    state[i] -= 0.1;
            }
        }else{
             for(size_t i = 0; i < state.size(); ++i){
                     state[i] += 0.1;
            }
        }

         if (stepCount >= maxSteps)
        {
          if (state[0] > 1.0)
            {
              reward = 10.0;
            }
           return std::make_tuple(state, reward, true);
        }

        return std::make_tuple(state, reward, false);
    }
    std::vector<double> getState() const{
        return state;
    }
};