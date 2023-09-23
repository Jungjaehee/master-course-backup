## DCUNET_ATT ##

 -- TIMIT 16K DATASET --

1. ctrl
  - 파일 리스트 파일
  1) timit_all.fileids : 데이터 전체 리스트 정보
  2) timit_test : 테스트 데이터
  3) timit_test_all : Test + Valid 데이터
  4) *_full_path : 해당 * 데이터 + 잡음 종류
  5) timit_valid_full_path_serve : valid 데이터 중 일부 추출

---
2. DCUNET_CA
  **  DCUNET / DCUNET with ATTENTION Models
  [ 실행 방법 ]
   1) config/config.yaml의 데이터 경로(target_dir, mixed_dir) 설정 
   2) run.sh 파일(step, model_type, output_model, data_dir, checkpoint_path, output_data_dir) 설정
     - model_type=DCUNET : DCUNET Model
     - model_type=ATTENTION : DCUNET with attention models
                                        (DCUNET-ATT, DCUNET-ATT+FD, DCUNET TFSA DE, SkipConv, SDAB)
       ** MODEL/model.py 중 class ComplexAttention 에서 추가 설정 필요 ** 
   
   3) run.sh 파일 실행
    - step=1 : Training    # 훈련 중지 때까지 훈련 진행됨
      * checkpoint model : chkpt/$output_model
      * log data : logs/$output_model (tensorboard 이용 확인 가능)
      
    - step=2 : Inference

--- 
3. Evaluate
  - SDR / PESQ / STOI evaluation
  [ 실행 방법 ]
  1) run.sh 파일 설정
    ( data_dir_name, base_dir, wav_path)
  
  2) run.sh 파일 실행 후 각 폴더에서 결과 확인 가능


---
4. UNET
  ** UNET Model **
  [ 실행 방법 ]
   1) config/config.yaml의 데이터 경로(target_dir, mixed_dir) 설정 
   2) run.sh 파일(step, model_type, data_dir, checkpoint_path, output_data_dir) 설정 후 실행

---
5. requirements
  - 설치 파일
  - pip install -r requirements.txt