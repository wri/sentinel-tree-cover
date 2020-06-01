# Easy Jupyter Deployment on ECS With Terraform
With this Terraform script, you can easily deploy a Jupyter Notebook on ECS. I want to refer to [this medium post](https://medium.com/@stephanschrijver/spinning-up-jupyter-notebooks-as-ecs-service-in-aws-with-terraform-805ac111d74b) for further explanation.   

## 1. Install terraform
First, install terraform: https://learn.hashicorp.com/terraform/getting-started/install.html

## 2. Init

Initialize terraform within this folder, this will download all dependencies et cetera used by this script.

```
terraform init
```

## 3. Apply
Apply the terraform scripts.

```
terraform apply -var-file=vars.tfvars
```

It will ask for a token, which you can use to connect the Jupyter Notebook, you can come up with one yourself.

```
Enter a value: secret-token
```
  
It will ask you to apply the changes, type 'yes'.

```
Enter a value: yes
```

The url which you can use will be displayed after the script successfully ran, it will take a minute of 5 to be up and running.


```
Apply complete! Resources: 8 added, 0 changed, 0 destroyed.

Outputs:

url = jupyter-4vkuf8x0.domainname.com?token=secret-token
```


## 4. Remove
When you're done with the jupyter notebook and you want to kill the environment, do:

```
terraform destroy -var-file=vars.tfvars
```

It will prompt for the token again, you should enter the token you used on applying.

