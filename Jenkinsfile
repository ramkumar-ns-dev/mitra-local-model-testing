pipeline {
    agent any

    parameters {
        string(
            name: 'HINDI_ASR_MODEL',
            defaultValue: 'vasista22/whisper-hindi-small',
            description: 'Hugging Face Model ID for Hindi Speech-to-Text (ASR)'
        )
        string(
            name: 'TAMIL_ASR_MODEL',
            defaultValue: 'vasista22/whisper-tamil-small',
            description: 'Hugging Face Model ID for Tamil Speech-to-Text (ASR)'
        )
        string(
            name: 'ENGLISH_ASR_MODEL',
            defaultValue: 'openai/whisper-tiny',
            description: 'Hugging Face Model ID for English Speech-to-Text (ASR)'
        )
        string(
            name: 'IMAGE_TAG',
            defaultValue: 'ramkumarnallusamy/airepositiorymitra-backend:1.0.4',
            description: 'Docker image and tag to deploy'
        )
        booleanParam(
            name: 'PULL_IMAGE',
            defaultValue: false,
            description: 'Pull the latest Docker image from registry before redeploying'
        )
    }

    environment {
        // Configure your server details and Jenkins credential ID here
        DEPLOY_HOST = 'your-server-ip-or-domain'
        DEPLOY_USER = 'ubuntu'
        PROJECT_DIR = '/path/to/AI repositiory mitra' // Path where docker-compose.yml lives on the server
        SSH_CRED_ID = 'mitra-server-ssh'              // Jenkins Credentials ID (SSH Username with private key)
    }

    stages {
        stage('Deploy to Server') {
            steps {
                sshagent([SSH_CRED_ID]) {
                    sh """
                    ssh -o StrictHostKeyChecking=no ${DEPLOY_USER}@${DEPLOY_HOST} << 'EOF'
                        set -e
                        cd "${PROJECT_DIR}"

                        echo "=== Current Directory: \$(pwd) ==="

                        # 1. Update ASR_MODELS_JSON configuration in docker-compose.yml / .env
                        python3 -c "
import json, re

hindi_model = '${params.HINDI_ASR_MODEL}'.strip()
tamil_model = '${params.TAMIL_ASR_MODEL}'.strip()
english_model = '${params.ENGLISH_ASR_MODEL}'.strip()

models_dict = {
    'ta': tamil_model,
    'hi': hindi_model,
    'en': english_model,
    'default': tamil_model
}

json_compact = json.dumps(models_dict)
json_formatted = json.dumps(models_dict, indent=10)

# Check and update docker-compose.yml
try:
    with open('docker-compose.yml', 'r') as f:
        compose_text = f.read()

    # Match ASR_MODELS_JSON block in docker-compose.yml
    pattern_compose = r'(ASR_MODELS_JSON:\s*>-?\s*\{[\s\S]*?\})'
    replacement_compose = f'ASR_MODELS_JSON: >-\\n          {json.dumps(models_dict, indent=10)}'

    if re.search(pattern_compose, compose_text):
        new_compose = re.sub(pattern_compose, replacement_compose, compose_text)
        with open('docker-compose.yml', 'w') as f:
            f.write(new_compose)
        print('Successfully updated ASR_MODELS_JSON in docker-compose.yml')
    else:
        print('ASR_MODELS_JSON block not found in docker-compose.yml; skipping file update.')
except Exception as e:
    print(f'Note on docker-compose.yml update: {e}')

# Check and update .env if present
try:
    with open('.env', 'r') as f:
        env_text = f.read()

    pattern_env = r\"ASR_MODELS_JSON='[\\s\\S]*?'\"
    replacement_env = f\"ASR_MODELS_JSON='{json.dumps(models_dict, indent=2)}'\"

    if re.search(pattern_env, env_text):
        new_env = re.sub(pattern_env, replacement_env, env_text)
        with open('.env', 'w') as f:
            f.write(new_env)
        print('Successfully updated ASR_MODELS_JSON in .env')
except FileNotFoundError:
    pass
except Exception as e:
    print(f'Note on .env update: {e}')
"

                        # 2. Update Image Tag in docker-compose.yml if provided
                        IMAGE="${params.IMAGE_TAG}"
                        if [ -n "\$IMAGE" ]; then
                            sed -i.bak -E "s|image: .*airepositiorymitra-backend:.*|image: \${IMAGE}|g" docker-compose.yml || true
                        fi

                        # 3. Pull new image if requested
                        if [ "${params.PULL_IMAGE}" = "true" ]; then
                            echo "Pulling Docker image: \${IMAGE}..."
                            docker compose pull airepositorymitra-backend || docker pull "\${IMAGE}"
                        fi

                        # 4. Redeploy container
                        echo "Restarting service with updated model configuration..."
                        docker compose up -d --remove-orphans airepositorymitra-backend

                        echo "Container status:"
                        docker compose ps
EOF
                    """
                }
            }
        }

        stage('Verify Health') {
            steps {
                echo "Waiting 10 seconds for backend to start up..."
                sleep 10
                sh """
                # Run health check against container (or via SSH on target server)
                ssh -o StrictHostKeyChecking=no ${DEPLOY_USER}@${DEPLOY_HOST} "curl -s -f http://127.0.0.1:3970/api/v1/health || curl -s -f http://127.0.0.1:3970/docs > /dev/null" && echo "Service is UP and Healthy!" || echo "Warning: Health check did not respond yet; verify logs with 'docker compose logs'."
                """
            }
        }
    }

    post {
        success {
            echo "Deployment completed successfully with Hindi ASR model: ${params.HINDI_ASR_MODEL}"
        }
        failure {
            echo "Deployment failed! Please check Jenkins console logs and server docker logs."
        }
    }
}
