"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useToast } from "@/components/ui/use-toast";
import { Loader2, Upload, Activity, ImageIcon, X, MoonIcon, SunIcon, Clock, Layers, Calculator, Shield, Download } from "lucide-react";
import { useTheme } from "next-themes";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { Progress } from "@/components/ui/progress";

interface Prediction {
  class_id: number;
  class_name: string;
  confidence: number;
  mask: number[][];  // segmentation mask coordinates
  area: number;
}

interface DayaniklilikAnalizi {
  kategori: string;
  puan: number;
  aciklama: string;
  toplam_hasar_orani: number;
  seviye1_orani: number;
  seviye2_orani: number;
}

interface ProcessResponse {
  filename: string;
  predictions: Prediction[];
  success: boolean;
  message: string;
  logs: string[];    // API logs
  image_size: {
    width: number;
    height: number;
  };
  dayaniklilik_analizi: DayaniklilikAnalizi;
  tile_count: number;
  processing_time: number;
  processed_image_base64: string | null;  // Processed image with damage visualization
}

// Dayanıklılık kategorisine göre renk seçimi
const getDurabilityColor = (kategori: string) => {
  switch (kategori) {
    case "mukemmel": return "text-green-600 bg-green-100";
    case "cok_iyi": return "text-green-500 bg-green-50";
    case "iyi": return "text-yellow-600 bg-yellow-100";
    case "orta": return "text-orange-600 bg-orange-100";
    case "zayif": return "text-red-500 bg-red-100";
    case "cok_zayif": return "text-red-700 bg-red-200";
    case "kritik": return "text-red-900 bg-red-300";
    default: return "text-gray-600 bg-gray-100";
  }
};

// Dayanıklılık puanına göre progress bar rengi
const getDurabilityProgressColor = (puan: number) => {
  if (puan >= 85) return "bg-green-500";
  if (puan >= 70) return "bg-yellow-500";
  if (puan >= 55) return "bg-orange-500";
  return "bg-red-500";
};

export default function Home() {
  const { setTheme, theme } = useTheme();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ProcessResponse | null>(null);
  const [health, setHealth] = useState<{ status: string; model_loaded: boolean } | null>(null);
  const [apiLogs, setApiLogs] = useState<string[]>([]);
  const { toast } = useToast();

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      setPreview(URL.createObjectURL(selectedFile));
      setResults(null); // Clear previous results including processed image
    }
  };

  const checkHealth = async () => {
    try {
      const response = await fetch("http://localhost:8000/health");
      const data = await response.json();
      setHealth(data);
    } catch (error) {
      console.error("Health check failed:", error);
      setHealth({ status: "error", model_loaded: false });
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      toast({
        title: "Dosya seçilmedi",
        description: "Lütfen işlenecek bir görsel seçin",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/process-image/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data: ProcessResponse = await response.json();
      setResults(data);
      setApiLogs(data.logs || []);

      // Dayanıklılık analizine göre toast mesajı
      const durability = data.dayaniklilik_analizi;
      let toastVariant: "default" | "destructive" = "default";
      if (durability.puan < 40) toastVariant = "destructive";

      toast({
        title: "İşlem tamamlandı",
        description: `${data.predictions.length} tespit bulundu | ${data.tile_count} parça işlendi | Dayanıklılık: ${durability.puan}/100`,
        variant: toastVariant,
      });
    } catch (error) {
      console.error("Görüntü işleme hatası:", error);
      toast({
        title: "Hata",
        description: "Görüntü işlenemedi",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const clearImage = () => {
    setFile(null);
    setPreview(null);
    setResults(null);
    setApiLogs([]);
  };

  const downloadProcessedImage = () => {
    if (results?.processed_image_base64) {
      const link = document.createElement('a');
      link.href = results.processed_image_base64;
      link.download = `processed_${results.filename}_damage_analysis.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      toast({
        title: "İndirme başladı",
        description: "İşlenmiş görsel indirilmeye başlandı",
      });
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-950 dark:to-slate-900">
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-between mb-8">
          <div className="space-y-1">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Gelişmiş Yapısal Analiz Sistemi
            </h1>
            <p className="text-muted-foreground">
              640x640 Tile İşleme | Dayanıklılık Analizi | Matematiksel Hesaplamalar
            </p>
          </div>

          <div className="flex items-center gap-4">
            <Button
              variant="outline"
              size="icon"
              onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
            >
              <SunIcon className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
              <MoonIcon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </div>

        {health && (
          <Alert className={`mb-6 ${health.status === "Healthy" ? "border-green-500 bg-green-50" : "border-red-500 bg-red-50"}`}>
            <Activity className="h-4 w-4" />
            <AlertTitle>
              {health.status === "Healthy" ? "✅ Sistem Aktif" : "❌ Sistem Hatası"}
            </AlertTitle>
            <AlertDescription>
              Model durumu: {health.model_loaded ? "Yüklendi ✓" : "Yüklenemedi ✗"}
            </AlertDescription>
          </Alert>
        )}

        <div className="grid gap-6 lg:grid-cols-3">
          {/* Sol Panel - Görsel Yükleme */}
          <Card className="lg:col-span-1">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <ImageIcon className="h-5 w-5" />
                  Görsel Yükle
                </CardTitle>
                <CardDescription>
                  Yapısal analiz için görsel yükleyin
                </CardDescription>
              </div>
              {preview && (
                <Button
                  onClick={handleSubmit}
                  disabled={loading}
                  size="sm"
                  className="h-8"
                >
                  {loading ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      İşleniyor
                    </>
                  ) : (
                    <>
                      <Upload className="mr-2 h-4 w-4" />
                      Analiz Et
                    </>
                  )}
                </Button>
              )}
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {!preview ? (
                  <div
                    className="border-2 border-dashed rounded-lg p-8 text-center hover:border-primary cursor-pointer transition-colors"
                    onClick={() => document.getElementById('file-upload')?.click()}
                  >
                    <div className="flex flex-col items-center gap-2">
                      <ImageIcon className="h-8 w-8 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">
                        Görüntüyü yüklemek için tıklayın
                      </p>
                      <p className="text-xs text-muted-foreground">
                        PNG, JPG veya JPEG (büyük görseller desteklenir)
                      </p>
                    </div>
                    <Input
                      id="file-upload"
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      className="hidden"
                    />
                  </div>
                ) : (
                  <div className="space-y-4">
                    <div className="relative">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="absolute top-2 right-2 rounded-full bg-background/80 backdrop-blur-sm z-10"
                        onClick={clearImage}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                      <div className="space-y-2">
                        <p className="text-sm font-medium text-muted-foreground">Orijinal Görsel</p>
                        <img
                          src={preview}
                          alt="Original Preview"
                          className="w-full h-auto rounded-lg border"
                        />
                      </div>
                    </div>

                    {/* İşlenmiş görsel karşılaştırması */}
                    {results && results.processed_image_base64 && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <p className="text-sm font-medium text-muted-foreground">İşlenmiş Görsel</p>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={downloadProcessedImage}
                            className="h-6 px-2 text-xs"
                          >
                            <Download className="mr-1 h-3 w-3" />
                            İndir
                          </Button>
                        </div>
                        <img
                          src={results.processed_image_base64}
                          alt="Processed with damage detection"
                          className="w-full h-auto rounded-lg border"
                        />
                        <div className="text-xs text-center text-muted-foreground bg-muted/50 rounded px-2 py-1">
                          {results.predictions.length} hasar bölgesi işaretlendi
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                            <span className="text-muted-foreground">Hafif</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-red-500 rounded-full"></div>
                            <span className="text-muted-foreground">Ağır</span>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Sağ Panel - Sonuçlar */}
          <div className="lg:col-span-2 space-y-6">
            {results && (
              <>
                {/* İşlenmiş Görsel - Hasar Tespiti Görselleştirmesi */}
                {results.processed_image_base64 && (
                  <Card>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle className="flex items-center gap-2">
                            <ImageIcon className="h-5 w-5" />
                            İşlenmiş Görsel - Hasar Tespiti
                          </CardTitle>
                          <CardDescription>
                            Tespit edilen hasarlar görsel üzerinde işaretlenmiştir
                          </CardDescription>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={downloadProcessedImage}
                        >
                          <Download className="mr-2 h-4 w-4" />
                          İndir
                        </Button>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="relative">
                        <img
                          src={results.processed_image_base64}
                          alt="Processed with damage detection"
                          className="w-full h-auto rounded-lg border shadow-lg"
                        />
                        <div className="absolute top-2 right-2 bg-black/60 text-white px-2 py-1 rounded text-xs">
                          {results.predictions.length} hasar tespit edildi
                        </div>
                      </div>
                      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 bg-yellow-400 rounded"></div>
                          <span>Seviye 1 Hasar (Hafif)</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-3 h-3 bg-red-500 rounded"></div>
                          <span>Seviye 2 Hasar (Ağır)</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )}

                {/* Dayanıklılık Analizi */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Dayanıklılık Analizi
                    </CardTitle>
                    <CardDescription>
                      Yapısal dayanıklılık değerlendirmesi ve hasar analizi
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Dayanıklılık Puanı</span>
                          <Badge className={getDurabilityColor(results.dayaniklilik_analizi.kategori)}>
                            {results.dayaniklilik_analizi.puan}/100
                          </Badge>
                        </div>
                        <Progress
                          value={results.dayaniklilik_analizi.puan}
                          className="h-2"
                        />
                        <div className={`h-2 rounded-full ${getDurabilityProgressColor(results.dayaniklilik_analizi.puan)}`}
                          style={{ width: `${results.dayaniklilik_analizi.puan}%` }} />
                      </div>

                      <div className="space-y-2">
                        <span className="text-sm text-muted-foreground">Kategori</span>
                        <Badge variant="outline" className={getDurabilityColor(results.dayaniklilik_analizi.kategori)}>
                          {results.dayaniklilik_analizi.kategori.replace('_', ' ').toUpperCase()}
                        </Badge>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-muted/50">
                      <p className="text-sm text-muted-foreground mb-2">Değerlendirme:</p>
                      <p className="text-sm">{results.dayaniklilik_analizi.aciklama}</p>
                    </div>

                    <div className="grid grid-cols-3 gap-4 text-center">
                      <div className="p-3 rounded-lg border">
                        <p className="text-2xl font-bold text-red-600">
                          %{results.dayaniklilik_analizi.toplam_hasar_orani}
                        </p>
                        <p className="text-xs text-muted-foreground">Toplam Hasar</p>
                      </div>
                      <div className="p-3 rounded-lg border">
                        <p className="text-2xl font-bold text-yellow-600">
                          %{results.dayaniklilik_analizi.seviye1_orani}
                        </p>
                        <p className="text-xs text-muted-foreground">Hafif Hasar</p>
                      </div>
                      <div className="p-3 rounded-lg border">
                        <p className="text-2xl font-bold text-red-600">
                          %{results.dayaniklilik_analizi.seviye2_orani}
                        </p>
                        <p className="text-xs text-muted-foreground">Ağır Hasar</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* İşlem Bilgileri */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Calculator className="h-5 w-5" />
                      İşlem Detayları
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <div className="flex items-center gap-2">
                        <Layers className="h-4 w-4 text-blue-500" />
                        <div>
                          <p className="text-sm text-muted-foreground">Tile Sayısı</p>
                          <p className="font-semibold">{results.tile_count}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-green-500" />
                        <div>
                          <p className="text-sm text-muted-foreground">İşlem Süresi</p>
                          <p className="font-semibold">{results.processing_time}s</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <ImageIcon className="h-4 w-4 text-purple-500" />
                        <div>
                          <p className="text-sm text-muted-foreground">Görsel Boyutu</p>
                          <p className="font-semibold">{results.image_size.width}x{results.image_size.height}</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Activity className="h-4 w-4 text-orange-500" />
                        <div>
                          <p className="text-sm text-muted-foreground">Tespit Sayısı</p>
                          <p className="font-semibold">{results.predictions.length}</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Tespit Sonuçları */}
                <Card>
                  <CardHeader>
                    <CardTitle>Tespit Sonuçları</CardTitle>
                    <CardDescription>
                      Görüntüde {results.predictions.length} hasar bölgesi tespit edildi
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {results.predictions.length > 0 ? (
                      <div className="space-y-2">
                        {results.predictions.map((pred, index) => (
                          <div key={index} className="flex items-center justify-between p-3 rounded-lg border">
                            <div className="flex items-center gap-3">
                              <Badge variant={pred.class_id === 0 ? "secondary" : "destructive"}>
                                {pred.class_name}
                              </Badge>
                              <span className="text-sm text-muted-foreground">
                                Güven: %{(pred.confidence * 100).toFixed(1)}
                              </span>
                            </div>
                            <span className="text-sm text-muted-foreground">
                              Alan: {Math.round(pred.area)} px²
                            </span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <Shield className="h-12 w-12 mx-auto mb-2 text-green-500" />
                        <p>Hasar tespit edilmedi</p>
                        <p className="text-sm">Yapı iyi durumda görünüyor</p>
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* API Logları */}
                <Card>
                  <CardHeader>
                    <CardTitle>İşlem Logları</CardTitle>
                    <CardDescription>Detaylı işlem adımları</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-muted rounded-lg p-4 max-h-48 overflow-y-auto">
                      <div className="space-y-1 font-mono text-xs">
                        {apiLogs.map((log, index) => (
                          <div key={index} className="text-muted-foreground">
                            {log}
                          </div>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </div>

        {/* Alt Bilgi */}
        <Card className="mt-8">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <h3 className="text-xl font-bold">Efe Hüseyin ÖZKAN</h3>
                <p className="text-sm text-muted-foreground">
                  YBS410 - Bitirme Projesi | Gelişmiş Yapısal Analiz Sistemi v2.0
                </p>
              </div>
              <div className="text-right space-y-1">
                <p className="text-sm text-muted-foreground">Özellikler:</p>
                <div className="flex gap-2">
                  <Badge variant="outline">640x640 Tile İşleme</Badge>
                  <Badge variant="outline">Dayanıklılık Analizi</Badge>
                  <Badge variant="outline">Matematiksel Hesaplamalar</Badge>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
