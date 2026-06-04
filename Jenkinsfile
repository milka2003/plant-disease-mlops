pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                git 'https://github.com/milka2003/plant-disease-mlops.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                echo 'Building Docker Image'
            }
        }

        stage('Test') {
            steps {
                echo 'Pipeline Working Successfully'
            }
        }
    }
}