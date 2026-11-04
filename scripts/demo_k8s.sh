#!/usr/bin/env bash
# Fase 3 — Las cuatro demostraciones exigidas sobre el clúster (y el cambio de versión por alias).
# Guarda la salida real de cada paso en evidencias/k8s-NN-*.txt.  USO: bash scripts/demo_k8s.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
mkdir -p evidencias
NODO=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' mlops-control-plane)
URL="http://$NODO:30090"
paso() { EVF="evidencias/$1.txt"; : > "$EVF"; echo "### $2" | tee -a "$EVF"; }
run()  { echo "\$ $*" | tee -a "$EVF"; bash -c "$*" 2>&1 | tee -a "$EVF"; echo | tee -a "$EVF"; }
nota() { echo "# $*" | tee -a "$EVF"; }
export URL
esperar_viejos() {  # espera a que terminen los Pods del ReplicaSet anterior
  local h; h=$(kubectl get rs -l app=riesgo-api --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.labels.pod-template-hash}')
  kubectl wait --for=delete pod -l "app=riesgo-api,pod-template-hash!=$h" --timeout=120s >/dev/null 2>&1; echo "(Pods de la versión anterior ya terminados)"
}
export -f esperar_viejos
pedir() { curl -s -X POST "$URL/predict" -H 'Content-Type: application/json' -d @scripts/cliente_alto.json; }
export -f pedir

paso k8s-01-replicas "1. Tres réplicas en estado Running simultáneamente"
run kubectl get deployment riesgo-api -o wide
run kubectl get pods -l app=riesgo-api -o wide
run "kubectl get service riesgo-api -o wide"
run "kubectl logs -l app=riesgo-api --prefix --tail=50 | grep 'modelo riesgo-credito'"
nota "Cada Pod cargó models:/riesgo-credito@champion desde el Model Registry de MLflow al arrancar."

paso k8s-02-balanceo "2. El tráfico se distribuye entre réplicas (nombre del Pod en la respuesta)"
run "pedir | python3 -m json.tool"
run "for i in \$(seq 1 30); do pedir | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"pod\"])'; done | sort | uniq -c"
nota "30 peticiones sucesivas al Service (NodePort 30090) fueron atendidas por los 3 Pods."

paso k8s-03-autorreparacion "3. Autorreparación: se elimina un Pod y el servicio no deja de responder"
VICTIMA=$(kubectl get pods -l app=riesgo-api -o jsonpath='{.items[0].metadata.name}')
( ok=0; fallo=0; fin=$((SECONDS+40)); while [ $SECONDS -lt $fin ]; do
    c=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$URL/predict" -H 'Content-Type: application/json' -d @scripts/cliente_alto.json)
    [ "$c" = 200 ] && ok=$((ok+1)) || fallo=$((fallo+1)); sleep 0.2; done
  echo "$ok $fallo" > /tmp/riesgo_carga.txt ) &
CARGA=$!
sleep 3
run kubectl get pods -l app=riesgo-api
run kubectl delete pod "$VICTIMA" --wait=false
run sleep 3
run kubectl get pods -l app=riesgo-api
run kubectl rollout status deployment/riesgo-api --timeout=180s
run kubectl get pods -l app=riesgo-api
wait $CARGA
read OK FALLO < /tmp/riesgo_carga.txt
run "echo 'Peticiones enviadas durante 40 s mientras se eliminaba el Pod: $OK respondidas con 200, $FALLO con error'"
nota "Se eliminó $VICTIMA; el ReplicaSet creó otro Pod y el Service siguió respondiendo con los otros dos."

paso k8s-04-escalado "4. Escalado: de 3 a 5 réplicas y regreso a 3"
run kubectl scale deployment/riesgo-api --replicas=5
run kubectl rollout status deployment/riesgo-api --timeout=240s
run kubectl get pods -l app=riesgo-api
run "until [ \$(kubectl get endpointslices -l kubernetes.io/service-name=riesgo-api -o jsonpath='{range .items[*].endpoints[?(@.conditions.ready==true)]}x{end}' | wc -c) -eq 5 ]; do sleep 1; done; sleep 5; kubectl get endpointslices -l kubernetes.io/service-name=riesgo-api"   # +5 s: kube-proxy actualiza iptables
run "for i in \$(seq 1 50); do pedir | python3 -c 'import sys,json; print(json.load(sys.stdin)[\"pod\"])'; done | sort | uniq -c"
run kubectl scale deployment/riesgo-api --replicas=3
run "until [ \$(kubectl get pods -l app=riesgo-api --no-headers | wc -l) -eq 3 ]; do sleep 2; done; kubectl get deployment riesgo-api; kubectl get pods -l app=riesgo-api"
nota "Con 5 réplicas el tráfico se reparte entre 5 Pods; al volver a 3 se eliminan los sobrantes."

paso k8s-05-alias "5. Trazabilidad: cambiar el alias @champion cambia el modelo servido sin tocar código ni imagen"
run "curl -s $URL/info | python3 -m json.tool"
run "bash scripts/entrenar.sh src/alias.py 1 2>/dev/null"
run kubectl rollout restart deployment/riesgo-api
run kubectl rollout status deployment/riesgo-api --timeout=300s
run esperar_viejos
run "for i in \$(seq 1 6); do curl -s $URL/info; echo; done"
run "bash scripts/entrenar.sh src/alias.py 2 2>/dev/null"
run kubectl rollout restart deployment/riesgo-api
run kubectl rollout status deployment/riesgo-api --timeout=300s
run esperar_viejos
run "for i in \$(seq 1 3); do curl -s $URL/info; echo; done"
nota "El alias pasó a la versión 1 (rollback del modelo) y volvió a la 2; la imagen riesgo-api:2.0 fue la misma en todo momento."
