package stinsoncerra_j.aslclassifier;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.ThumbnailUtils;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextSwitcher;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;
import java.util.Collections;


public class MainActivity extends AppCompatActivity {//implements SurfaceHolder.Callback, Handler.Callback {

    static final String TAG = "Camera";
    static final int PERMISSIONS_REQUEST_CAMERA = 1242;
    private static final int MSG_CAMERA_OPENED = 1;
    private static final int MSG_SURFACE_READY = 2;

   // private final Handler mHandler = new Handler(this);
    TextureView textureView;
   // SurfaceHolder mSurfaceHolder;
    TextureView.SurfaceTextureListener mSurfaceTextureListener;
    CameraManager mCameraManager;
    int cameraFacing;
    Size previewSize;
    String cameraId;

    HandlerThread backgroundThread;
    Handler backgroundHandler;

    CameraDevice.StateCallback stateCallback;
    CameraDevice cameraDevice;
    CameraCaptureSession cameraCaptureSession;
    CaptureRequest.Builder captureRequestBuilder;
    CaptureRequest captureRequest;

    TensorFlowInferenceInterface tensorFlowInferenceInterface;


    //String[] mCameraIDsList;
    //CameraDevice.StateCallback mCameraStateCB;
   // CameraDevice mCameraDevice;
   // CameraCaptureSession mCaptureSession;
  //  boolean mSurfaceCreated = true;
   // boolean mIsCameraConfigured = false;
   // private Surface mCameraSurface = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ActivityCompat.requestPermissions(this, new String[] {Manifest.permission.CAMERA}, PERMISSIONS_REQUEST_CAMERA);

        mCameraManager = (CameraManager) getSystemService(Context.CAMERA_SERVICE);
        cameraFacing = CameraCharacteristics.LENS_FACING_BACK;

        this.textureView = (TextureView) findViewById(R.id.textureView);

        tensorFlowInferenceInterface = new TensorFlowInferenceInterface(getAssets(), "opt_asl_convnet.pb");

        mSurfaceTextureListener = new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture surfaceTexture, int i, int i1) {
                setUpCamera();
                openCamera();
            }

            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture surfaceTexture, int i, int i1) {

            }

            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture surfaceTexture) {
                return false;
            }

            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture surfaceTexture) {

            }
        };

        stateCallback = new CameraDevice.StateCallback(){

            @Override
            public void onOpened(@NonNull CameraDevice cameraDevice){
                MainActivity.this.cameraDevice = cameraDevice;
                createPreviewSession();
            }

            @Override
            public void onDisconnected(@NonNull CameraDevice cameraDevice) {
                cameraDevice.close();
            }

            @Override
            public void onError(@NonNull CameraDevice cameraDevice, int i) {
                cameraDevice.close();
                MainActivity.this.cameraDevice = null;
            }
        };
        //this.mSurfaceHolder = this.mSurfaceView.getHolder();
        //this.mSurfaceHolder.addCallback(this);
        /*this.mCameraManager = (CameraManager) this.getSystemService(Context.CAMERA_SERVICE);

        try {
            mCameraIDsList = this.mCameraManager.getCameraIdList();
            for (String id: mCameraIDsList){
                Log.v(TAG, "CameraID: " + id);
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }*/
/*
        mCameraStateCB = new CameraDevice.StateCallback() {
            @Override
            public void onOpened(@NonNull CameraDevice cameraDevice) {
                Toast.makeText(getApplicationContext(), "onOpened", Toast.LENGTH_SHORT).show();

                mCameraDevice = cameraDevice;
                mHandler.sendEmptyMessage(MSG_CAMERA_OPENED);
            }

            @Override
            public void onDisconnected(@NonNull CameraDevice cameraDevice) {
                Toast.makeText(getApplicationContext(), "onDisconnected", Toast.LENGTH_SHORT).show();
            }

            @Override
            public void onError(@NonNull CameraDevice cameraDevice, int i) {
                Toast.makeText(getApplicationContext(), "onError", Toast.LENGTH_SHORT).show();
            }
        };
*/
        final TextView text = (TextView) findViewById(R.id.textView);
        Button recognizeButton = (Button)findViewById(R.id.btnRecognize);



        recognizeButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                /*Bitmap bitmap = textureView.getBitmap(50, 50);
                int[] labelList = new int[26];
                for(int i = 0; i < labelList.length; i++){
                    labelList[i] = i;
                }
                float[][] labelProbArray = new float[1][26];
                int size = bitmap.getRowBytes() * bitmap.getHeight();
                ByteBuffer byteBuffer = ByteBuffer.allocate(size);
                bitmap.copyPixelsToBuffer(byteBuffer);
                tflite.run(byteBuffer.array(), labelProbArray);*/
                //"dense_2/Softmax"
                float[] pixels = getPixels(textureView.getBitmap(50, 50));
                tensorFlowInferenceInterface.feed("conv2d_1_input", pixels,  1, 50, 50, 3);
                //String[] outputnames = new String[] { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25" };//"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z" };
                String[] outputnames = new String[] { "dense_2/Softmax" };
                tensorFlowInferenceInterface.run(outputnames);
                float[] output = new float[26];
                tensorFlowInferenceInterface.fetch("dense_2/Softmax", output);

                float highestProb = 0;
                char className = 0;
                for(int i = 0; i < output.length; i++){
                    Log.v(TAG, "Class: " + i + " Prob: " + output[i]);
                    if(output[i] > highestProb){
                        highestProb = output[i];
                        //Log.v(TAG, "Class: " + i + " Prob: " + output[i]);
                        className = getChar(i);
                    }
                }
                if(highestProb > 0.3){
                    text.setText("Prediction: " + className + " Probability:" + highestProb);
                    // Toast.makeText(getApplicationContext(), className, Toast.LENGTH_LONG).show();

                }

            }
        });

        /*try {
            tflite = new Interpreter(loadModelFile(this));
        }catch(IOException e){
            e.printStackTrace();
            Toast.makeText(getApplicationContext(), "Error opening model file", Toast.LENGTH_SHORT).show();
            return;
        }*/

    }
/*
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd("file:///android_asset/opt_asl_convnet.pb");
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }*/

    private char getChar(int classVal){
        return (char)('A' + classVal);
    }

    private static float[] getPixels(Bitmap bitmap){
        int[] intValues = new int[50 * 50];
        float[] floatValues = new float[50 * 50 * 3];
        if(bitmap.getHeight() != 50 || bitmap.getWidth() != 50){
            bitmap = ThumbnailUtils.extractThumbnail(bitmap, 50, 50);
        }
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for(int i = 0; i < intValues.length; i++){
            final int val = intValues[i];
            floatValues[i * 3 + 2] = Color.red(val);
            floatValues[i * 3 + 1] = Color.green(val);
            floatValues[i * 3] = Color.blue(val);
            //int pix = intValues[i];
            //int b = pix & 0xff;
            //floatValues[i] = 0xff - b;
        }
        return floatValues;
    }

    private void setUpCamera(){
        try {
            for (String id: mCameraManager.getCameraIdList()){
                Log.v(TAG, "CameraID: " + id);
                CameraCharacteristics cameraCharacteristics = mCameraManager.getCameraCharacteristics(id);
                if(cameraCharacteristics.get(CameraCharacteristics.LENS_FACING) == cameraFacing){
                    StreamConfigurationMap streamConfigurationMap = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                    previewSize = streamConfigurationMap.getOutputSizes(SurfaceTexture.class)[0];
                    this.cameraId = id;
                }
            }
        } catch (CameraAccessException e){
            e.printStackTrace();
        }
    }

    private void openCamera() {
        try {
            if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
                mCameraManager.openCamera(cameraId, stateCallback, backgroundHandler);
            }
        }catch (CameraAccessException e){
            e.printStackTrace();
        }
    }

    private void openBackgroundThread(){
        backgroundThread = new HandlerThread("camera_background_thread");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    @Override
    protected void onResume() {
        super.onResume();
        openBackgroundThread();
        if (textureView.isAvailable()) {
            setUpCamera();
            openCamera();
        } else {
            textureView.setSurfaceTextureListener(mSurfaceTextureListener);
        }
    }

    @Override
    protected void onStop(){
        super.onStop();
        closeCamera();
        closeBackgroundThread();
    }

    private void closeCamera() {
        if(cameraCaptureSession != null){
            cameraCaptureSession.close();
            cameraCaptureSession = null;
        }

        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
    }

    private void closeBackgroundThread() {
        if (backgroundHandler != null) {
            backgroundThread.quitSafely();
            backgroundThread = null;
            backgroundHandler = null;
        }
    }

    private void createPreviewSession() {
        try {
            SurfaceTexture surfaceTexture = textureView.getSurfaceTexture();
            surfaceTexture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
            Surface previewSurface = new Surface(surfaceTexture);
            captureRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            captureRequestBuilder.addTarget(previewSurface);

            cameraDevice.createCaptureSession(Collections.singletonList(previewSurface),
                    new CameraCaptureSession.StateCallback() {

                        @Override
                        public void onConfigured(CameraCaptureSession cameraCaptureSession) {
                            if (cameraDevice == null) {
                                return;
                            }

                            try {
                                captureRequest = captureRequestBuilder.build();
                                MainActivity.this.cameraCaptureSession = cameraCaptureSession;
                                MainActivity.this.cameraCaptureSession.setRepeatingRequest(captureRequest,
                                        null, backgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession cameraCaptureSession) {

                        }
                    }, backgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }
/*
    @Override
    protected void onStart() {
        super.onStart();

        int permissionCheck = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA);
        //if(permissionCheck != PackageManager.PERMISSION_GRANTED){
            if(ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.CAMERA)){

            } else {
                ActivityCompat.requestPermissions(this, new String[]{ Manifest.permission.CAMERA }, PERMISSIONS_REQUEST_CAMERA);
                Toast.makeText(getApplicationContext(), "request permision", Toast.LENGTH_SHORT).show();
            }
       /* } else {
            Toast.makeText(getApplicationContext(), "PERMISSION_ALREADY_GRANTED", Toast.LENGTH_SHORT).show();
            try {
                mCameraManager.openCamera(mCameraIDsList[0], mCameraStateCB, new Handler());
            } catch (CameraAccessException e){
                e.printStackTrace();
            }
        }*/
/*
    }

    @Override
    protected void onStop(){
        super.onStop();
        try {
            if(mCaptureSession != null){
                mCaptureSession.stopRepeating();
                mCaptureSession.close();
                mCaptureSession = null;
            }
            mIsCameraConfigured = true;
        } catch (final CameraAccessException e){
            e.printStackTrace();
        } catch (final IllegalStateException e2) {
            e2.printStackTrace();
        } finally {
            if(mCameraDevice != null){
                mCameraDevice.close();
                mCameraDevice = null;
                mCaptureSession = null;
            }
        }
    }

    @Override
    public boolean handleMessage(Message msg) {
        switch(msg.what){
            case MSG_CAMERA_OPENED:
            case MSG_SURFACE_READY:
                if(mSurfaceCreated && (mCameraDevice != null)){
                    configureCamera();
                }
                break;
        }
        return true;
    }

    private void configureCamera() {
        // prepare list of surfaces to be used in capture requests
        List<Surface> sfl = new ArrayList<Surface>();

        sfl.add(mCameraSurface); // surface for viewfinder preview

        // configure camera with all the surfaces to be ever used
        try {
            mCameraDevice.createCaptureSession(sfl,
                    new CaptureSessionListener(), null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }

        mIsCameraConfigured = true;
    }


    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode) {
            case PERMISSIONS_REQUEST_CAMERA:
                if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
                    try {
                        if(grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                            mCameraManager.openCamera(mCameraIDsList[0], mCameraStateCB, new Handler());
                        }
                    } catch (CameraAccessException e) {
                        e.printStackTrace();
                    }
                break;
        }
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        mCameraSurface = holder.getSurface();
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        mCameraSurface = holder.getSurface();
        mSurfaceCreated = true;
        mHandler.sendEmptyMessage(MSG_SURFACE_READY);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        mSurfaceCreated = false;
    }

    private class CaptureSessionListener extends
            CameraCaptureSession.StateCallback {
        @Override
        public void onConfigureFailed(final CameraCaptureSession session) {
            Log.d(TAG, "CaptureSessionConfigure failed");
        }

        @Override
        public void onConfigured(final CameraCaptureSession session) {
            Log.d(TAG, "CaptureSessionConfigure onConfigured");
            mCaptureSession = session;

            try {
                CaptureRequest.Builder previewRequestBuilder = mCameraDevice
                        .createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                previewRequestBuilder.addTarget(mCameraSurface);
                mCaptureSession.setRepeatingRequest(previewRequestBuilder.build(),
                        null, null);
            } catch (CameraAccessException e) {
                Log.d(TAG, "setting up preview failed");
                e.printStackTrace();
            }
        }
    }
    */
}
