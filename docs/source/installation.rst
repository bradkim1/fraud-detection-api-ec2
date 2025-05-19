Installation
===========

Prerequisites
------------

- Python 3.8+
- pip
- Docker (optional)

Local Installation
-----------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/bradkim1/fraud-detection-api-ec2.git
      cd fraud-detection-api-ec2

2. Install dependencies:

   .. code-block:: bash

      pip install -r requirements.txt

3. Run the API:

   .. code-block:: bash

      python enhanced_fraud_api.py

4. Access the UI:

   Open your browser and navigate to:
   
   .. code-block:: text
   
      http://localhost/

Docker Installation
------------------

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/bradkim1/fraud-detection-api-ec2.git
      cd fraud-detection-api-ec2

2. Build and run with Docker Compose:

   .. code-block:: bash

      docker-compose up -d

3. Access the UI:

   Open your browser and navigate to:
   
   .. code-block:: text
   
      http://localhost/

EC2 Deployment
-------------

1. Launch an EC2 instance with Ubuntu:

   - Choose Ubuntu Server 20.04 or later
   - Ensure ports 22 (SSH), 80 (HTTP), and 8015 (API) are open in the security group

2. SSH into your instance:

   .. code-block:: bash

      ssh -i your-key.pem ubuntu@your-ec2-ip

3. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/bradkim1/fraud-detection-api-ec2.git
      cd fraud-detection-api-ec2

4. Install dependencies:

   .. code-block:: bash

      sudo apt-get update
      sudo apt-get install -y python3-pip nginx
      pip install -r requirements.txt

5. Set up Nginx:

   .. code-block:: bash

      sudo cp nginx/fraud-ui.conf /etc/nginx/sites-available/
      sudo ln -s /etc/nginx/sites-available/fraud-ui.conf /etc/nginx/sites-enabled/
      sudo rm /etc/nginx/sites-enabled/default
      sudo systemctl restart nginx

6. Create and start the API service:

   .. code-block:: bash

      sudo cp systemd/fraud-api.service /etc/systemd/system/
      sudo systemctl enable fraud-api
      sudo systemctl start fraud-api

7. Access the UI:

   Open your browser and navigate to:
   
   .. code-block:: text
   
      http://your-ec2-ip/
