# Deployment

`compose.prod.yml` runs the API behind [Caddy](https://caddyserver.com/), which gets and renews a Let's Encrypt certificate for `DOMAIN`. The API container has no public port, a 512 MB memory limit and a read-only filesystem.

Requirements on the server: Docker with the compose plugin, ports 80 and 443 free, and a DNS A record for `DOMAIN` pointing at the server.

```bash
git clone https://github.com/eigen-ml/2d-magnetic-materials-ml.git /opt/2dml
cd /opt/2dml
DOMAIN=2dml.example.com docker compose -f deploy/compose.prod.yml up -d --build
curl https://2dml.example.com/health
```

Update: `git pull` and run the same `up -d --build` command. Stop: `docker compose -f deploy/compose.prod.yml down`.

If another web server already uses ports 80/443 on the host, drop the `caddy` service, publish the API on `127.0.0.1:8000` and add a reverse-proxy entry for it in the existing server instead.
