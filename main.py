import cv2 

# capture frame dari video yang kita input
cap = cv2.VideoCapture('video.avi') 
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH) #deteksi lebar frame video

#DEFINISI atribut untuk membuat garis (detector)
lineVerticalPosition = 100 #definisi tinggi garis pada video
startPoint = (0, lineVerticalPosition) #definisi kordinat titik awal garis
endPoint = (int(frameWidth), lineVerticalPosition) #definisi kordinat titik akhir garis
lineColor = (255, 0, 0) #definisi warna garis
thickNess = 2 #definisi ketebalan garis

# cars.xml merupakan hasil training dari klasifikasi mobil sebagai objek yang akan di deteksi
car_cascade = cv2.CascadeClassifier('cars.xml')

while True: 
    ret, frames = cap.read() #frame yang di capture mulai di baca
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY) #warna dasar video diubah ke grayscale setiap frame

    #start membuat garis
    cv2.line(frames, startPoint, endPoint, lineColor, thickNess)

    cars = car_cascade.detectMultiScale(gray, 1.1, 1) # deteksi mobil

    #definisi atribut pembuatan frame
    carFrameColor = (0,0,255)
    
    violators = 0 #variabel pendeteksi pelanggar

	# objek yang terdeteksi sebagai mobil di buat frame disekitarnya
    for (x,y,w,h) in cars: 
        cv2.rectangle(frames, (x,y), (x+w,y+h), carFrameColor, thickNess)
        
        # if(y > lineVerticalPosition){
        #     violators ++
        # }

    # menampilkan hasil capture video + objek yang sudah dibuat frame disekitarnya
    cv2.imshow('video2', frames)

    # program terus dijalankan hingga user tekan tombol ESC
    if cv2.waitKey(33) == 27: 
        break

# De-alokasi semua penggunaan memori
cv2.destroyAllWindows() 
