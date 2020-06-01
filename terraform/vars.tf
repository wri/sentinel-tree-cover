variable "vpc_id" {}
variable "region" {}
variable "profile_name" {}
variable "ecs_cluster_name" {}
variable "loadbalancer_arn" {}
variable "hosted_zone_id" {}
variable "fargate_subnets" {
  type = list(string)
}
variable "token" {}
variable "jupyter_docker_tag" {}
variable "memory" {}
variable "cpu" {}
variable "domain" {}
