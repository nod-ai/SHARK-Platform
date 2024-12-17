# LLama 8b GPU instructions on Kubernetes

## Setup

We will use an example with `llama_8b_f16` in order to describe the
process of exporting a model and deploying four instances of a shortfin llm server
behind a load balancer on MI300X GPU.

### Pre-Requisites

- Kubernetes cluster available to use
- kubectl installed on system and configured for cluster of interest
    - To install kubectl, please check out [kubectl install](https://kubernetes.io/docs/tasks/tools/#kubectl)
    and make sure to set the `KUBECONFIG` environment variable to point to your kube config file to authorize
    connection to the cluster.

### Deploy shortfin llama app service

Please edit the following file to fetch the correct artifacts and serve the intended configuration of the llama3 model for your use case [here](https://github.com/nod-ai/shark-ai/tree/main/shortfin/python/shortfin_apps/llm/k8s/llama-app-deployment.yaml).

To deploy llama app:

```
kubectl apply -f llama-app-deployment.yaml
```

To retrieve external IP for targetting the llama app load balancer:

```
kubectl get service shark-llama-app-service
```

Now, you can use the external IP for sglang integration or just sending image generation requests.

### Delete shortfin llama app service

After done using, make sure to delete:

```
kubectl delete deployment shark-llama-app-deployment
kubectl delete service shark-llama-app-service
```
