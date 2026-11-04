# Uso interno (source): define NODO, la dirección desde la que se alcanzan los NodePort del clúster.
#  - Si el clúster se creó con k8s/kind-config.yaml, kind publica 30090/30091 en localhost -> 127.0.0.1 (funciona en todos los SO).
#  - Si no, se usa la IP del contenedor del nodo (solo es alcanzable desde el anfitrión en Linux).
if docker port mlops-control-plane 30090/tcp >/dev/null 2>&1; then
  NODO=127.0.0.1
else
  NODO=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlops-control-plane)
fi
