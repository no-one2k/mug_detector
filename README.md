# mug_detector
opencv based detector of coffee mug wrapped in dockerized web-service

Что сделано:
1. скрипт по детекции кружки в 2 вариантах
  - простое выделение по цвету (кружка = синий цвет): `python3 app/script1.py -i data/2018-02-2715_03_24.ogv -d blue`
  - детекция с помощью yolo (через opencv): `python3 app/script1.py -i data/2018-02-2715_03_24.ogv -d yolo` 
2. простой сервис на [Falcon](https://falconframework.org), который при запросе возвращает html-страницу с моментами, когда кружка выходит из кадра и появляется
3. docker образ из alpine + python3.7 + opencv4.1 + falcon


Что не сделано:
1. детекция текста. В голове рассмотрены варианты из
  - opencv методов по выделению объекта и фона
  - tesseract 
  - [EAST через opencv](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)
2. unit test'ы
3. документация и комментарии


Как запускать
1. `git clone`
2. добавить 2018-02-2715_03_24.ogv внутрь папки `data/`
3. собрать образ `docker build -t mug_detector:1.0 .`
4. запустить `docker run -ti -p 7777:7777 -v $PWD/data:/data mug_detector:1.0`
5. открыть в браузере http://localhost:7777
